import random
import math
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller
import time
import aiohttp
from aiohttp_socks import SocksConnector, SocksVer
import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

pages_id_q = [i for i in range(3)]
random.shuffle(pages_id_q)
pages_q = []
K = 1
connections = None
ages = [time.time() for _ in range(K)]
locks = [False for _ in range(K)]
restaurant_tasks = 0
reviews_tasks = 0


async def async_session():
    with Controller.from_port(port=9051) as controller:
        controller.authenticate(password='')
        controller.signal(Signal.NEWNYM)
    connector = SocksConnector.from_url('socks5://127.0.0.1:9050')
    return await aiohttp.ClientSession(connector=connector).__aenter__(), connector


async def refresh():
    idx = random.randint(0, K - 1)
    while locks[idx]:
        idx = (idx + 1) % K
    locks[idx] = True
    if time.time() - ages[idx] > 320:
        print('*********************')
        await connections[idx][1].close()
        await connections[idx][0].__aexit__(None, None, None)
        connections[idx] = await async_session()
        ages[idx] = time.time()
    return idx


async def retrieve_restaurant_task():
    global restaurant_tasks
    if not pages_id_q:
        return
    page = pages_id_q.pop()
    print(page)
    idx = await refresh()
    response = await connections[idx][0].get(f'https://snappfood.ir/restaurant/?page={page}')
    soup = BeautifulSoup(await response.read(), 'lxml')
    soup = soup.find('div', id='main-container').find('div', class_='vendor-list-wrapper')
    names = set()
    for div in soup.findAll('div'):
        h2 = div.find('h2', class_='vendor-title')
        if h2:
            a = h2.find('a')
            _id = a['href'].split('/')
            if a and _id:
                title = a.decode_contents()
                if title in names:
                    continue
                names.add(title)
                _id = _id[_id.index('menu') + 1]
                reviews_size = div.find('li', class_='vendor-comments-btn').decode_contents()
                reviews_size = reviews_size = int(''.join([{
                                                               '۰': '0',
                                                               '۱': '1',
                                                               '۲': '2',
                                                               '۳': '3',
                                                               '۴': '4',
                                                               '۵': '5',
                                                               '۶': '6',
                                                               '۷': '7',
                                                               '۸': '8',
                                                               '۹': '9',
                                                           }.get(c, '') for c in reviews_size]))
                pages_q.append({'title': title, '_id': _id, 'pages': [i for i in range(math.ceil(reviews_size / 10))]})
    locks[idx] = False
    restaurant_tasks -= 1


async def restaurant_loop():
    global restaurant_tasks
    while pages_id_q:
        if restaurant_tasks + reviews_tasks < K:
            restaurant_tasks += 1
            asyncio.ensure_future(retrieve_restaurant_task())
        else:
            await asyncio.sleep(.001)


async def retrieve_review_task():
    global reviews_tasks
    if not pages_q:
        return
    vendor = pages_q[-1]
    idx = await refresh()
    if not vendor['pages']:
        pages_q.pop()
    else:
        page = vendor["pages"].pop()
        print(vendor['_id'], page)
        response = await connections[idx][0].get(f'https://snappfood.ir/restaurant/comment/vendor/{vendor["_id"]}/{page}')
        response = await response.json()
        comments = response['data']['comments']
        if not comments:
            if pages_q:
                pages_q.pop()
        else:
            print(comments)
    locks[idx] = False
    reviews_tasks -= 1


async def reviews_loop():
    global reviews_tasks
    while pages_q or pages_id_q or restaurant_tasks:
        if reviews_tasks + restaurant_tasks < K and pages_q:
            reviews_tasks += 1
            asyncio.ensure_future(retrieve_review_task())
        else:
            await asyncio.sleep(.001)


async def main():
    global connections
    connections = [await async_session() for _ in range(K)]
    asyncio.ensure_future(restaurant_loop())
    await asyncio.gather(restaurant_loop(), reviews_loop())
    print(pages_q)
    for sess, conn in connections:
        await conn.close()
        await sess.__aexit__(None, None, None)


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())
