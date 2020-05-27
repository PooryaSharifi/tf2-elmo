import random
import math
from bs4 import BeautifulSoup
from stem import Signal
from stem.control import Controller
import time
import aiohttp
from aiohttp_socks import SocksConnector, SocksVer
from multiprocessing import Process
import asyncio
import uvloop
import motor.motor_asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

reviews = None
pages_id_q = None
reviews_id_q = []
K = 16
connections = None
ages = [time.time() for _ in range(K)]
locks = [False for _ in range(K)]
tasks = 0


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
    if time.time() - ages[idx] > 1:
        await connections[idx][1].close()
        await connections[idx][0].__aexit__(None, None, None)
        connections[idx] = await async_session()
        ages[idx] = time.time()
    return idx


async def retrieve_restaurant_task():
    global tasks
    if not pages_id_q:
        tasks -= 1
        return
    page = pages_id_q.pop()
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
                print(_id, title)
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
                # for i in range(math.ceil(reviews_size / 10)):
                reviews_id_q.append({'vendor_title': title, 'vendor_id': _id, 'page': 0})
                reviews_id_q.append({'vendor_title': title, 'vendor_id': _id, 'page': 1})
    locks[idx] = False
    tasks -= 1


async def restaurant_loop():
    global tasks
    while pages_id_q or tasks:
        if tasks < K and pages_id_q:
            tasks += 1
            asyncio.ensure_future(retrieve_restaurant_task())
        else:
            await asyncio.sleep(.001)


async def retrieve_review_task():
    global tasks
    if not reviews_id_q:
        tasks -= 1
        return
    review_page = reviews_id_q.pop()
    idx = await refresh()
    vendor_id = review_page["vendor_id"]
    try:
        response = await connections[idx][0].get(f'https://snappfood.ir/restaurant/comment/vendor/{vendor_id}/{review_page["page"]}')
        response = await response.json()
        comments = response['data']['comments']
        if comments:
            for c in comments:
                c['vendor_id'] = vendor_id
                c['vendor_title'] = review_page['vendor_title']
            await reviews.insert_many(comments)
            if len(comments) == 10:
                reviews_id_q.append({'vendor_title': review_page['vendor_title'], 'vendor_id': vendor_id,
                                     'page': review_page['page'] + 2})
    except:
        reviews_id_q.append(review_page)
    locks[idx] = False
    tasks -= 1


async def reviews_loop():
    global tasks
    while reviews_id_q or tasks:
        if tasks < K and reviews_id_q:
            tasks += 1
            asyncio.ensure_future(retrieve_review_task())
        else:
            await asyncio.sleep(.001)


a = {'commentId': 8414317, 'date': '۱۳۹۸/۰۹/۱۴', 'createdDate': '2019-12-05 13:09:18', 'sender': 'جعفر',
     'customerId': '1115177', 'commentText': 'با سلام \nمثل همیشه خوب بود. \nفقط اینبار کمی تند شده بود. \nبا تشکر ',
     'rate': 10, 'feeling': 'HAPPY', 'status': 1, 'expeditionType': 'ZF_EXPRESS',
     'foods': [{'title': 'چلو خورشت قورمه سبزی'}, {'title': 'قاشق و چنگال'}], 'replies': []}


async def main():
    global connections, reviews_id_q, reviews
    reviews = motor.motor_asyncio.AsyncIOMotorClient('localhost', 27017)['snappfood']['reviews']
    connections = [await async_session() for _ in range(K)]
    asyncio.ensure_future(restaurant_loop())
    await restaurant_loop()
    random.shuffle(reviews_id_q)
    await reviews_loop()
    for sess, conn in connections:
        await conn.close()
        await sess.__aexit__(None, None, None)


def proc(skip, limit):
    global pages_id_q
    pages_id_q = [i for i in range(skip, skip + limit)]
    random.shuffle(pages_id_q)
    n = time.time()
    asyncio.get_event_loop().run_until_complete(main())
    print(f'time cost is {time.time() - n}')


if __name__ == '__main__':
    for i in range(16):
        Process(target=proc, args=(750 + i * 5, 5)).start()
