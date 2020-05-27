import pandas as pd
import codecs
import csv
import time

# reviews = pd.read_csv('_snappfood_reviews.csv')
# print(len(reviews))
# print(type(reviews['commentText'][0]))
#
# ids = reviews['commentId']
# bag = set()
# dl = []
# for i, _id in enumerate(ids.to_numpy()):
#     if _id in bag:
#         dl.append(i)
#     bag.add(_id)
#
# texts = reviews['commentText']
# bag = set()
# for i, text in enumerate(texts):
#     if text is float:
#         dl.append(i)
#
# reviews = reviews.drop(dl)
# reviews.to_csv("_snappfood_reviews.csv")

# texts = reviews['commentText'].to_numpy()
# print(texts[0])
# print(texts[471234])
# with open('snappfood_comments', 'w') as f:
#     f.write('\n'.join(texts))


# with open('snappfood.csv', 'rU', encoding='utf-8') as f:
#     with open('_snappfood.csv', 'w', encoding='utf-8') as g:
#         reader = csv.reader(x.replace('\0', '') for x in f)
#         writer = csv.writer(g)
#         ids = set()
#         d = 0
#         for line in reader:
#             if line[0] not in ids:
#                 ids.add(line[0])
#                 if line[3] != '' and line[2] != '' and line[2] != 'NOTHING':
#                     writer.writerow(line)
#         print(d)


# with open('_snappfood.csv', 'rU', encoding='utf-8') as f:
#     reader = csv.reader(f)
#     ids = set()
#     feelings = set()
#     rates = set()
#     nothings = set()
#     happies = set()
#     sads = set()
#     d = 0
#     nd = 0
#     for line in reader:
#         d += 1
#         if line[2] == 'NOTHING':
#             nothings.add(line[1])
#         if line[2] == 'SAD':
#             sads.add(line[1])
#         if line[2] == 'HAPPY':
#             happies.add(line[1])
#         if line[0] in ids:
#             print('gotcha')
#         ids.add(line[0])
#         feelings.add(line[2])
#         rates.add(line[1])
#     print(feelings)
#     print(rates)
#     print(nothings)
#     print(sads)
#     print(happies)
#     print(f'rows are {d - 1}')
#     print(nd)

# reviews = pd.read_csv('_snappfood.csv')
# with open('snappfood.txt', 'w', encoding='utf-8') as w:
#     w.write('\n'.join(reviews['commentText'].values))


import sentencepiece as spm
# spm.SentencePieceTrainer.Train('--input=snappfood.txt --model_prefix=snappfood_pieces --vocab_size=8192 --character_coverage=1.0')

sp = spm.SentencePieceProcessor()
sp.Load("snappfood_pieces.model")

# ids = sp.EncodeAsIds("مامانی بابایی دوست دارم من")
#
# print(ids)
# for _id in ids:
#     print(sp.decode_ids([_id]))


# reviews = pd.read_csv('_snappfood.csv')
# print(set(reviews['feeling'].values))
# print(len([x for x in reviews['feeling']]))
# _max = 0
# n = time.time()
# # print(max(len(sp.encode_as_ids(txt)) for txt in reviews['commentText'].values))
# d = {}
# for txt in reviews['commentText'].values:
#     l = len(sp.encode_as_ids(txt))
#     if l not in d:
#         d[l] = 0
#     d[l] += 1
# d = [(key, value) for key, value in d.items()]
# d = sorted(d, key=lambda x: x[0])
# print(d)
# print(zip(*d))
# print(time.time() - n)
# turn inputs and ouputs, split train test, model, compile


# d = [(0, 2), (1, 16416), (2, 27618), (3, 34693), (4, 30039), (5, 27455), (6, 25376), (7, 24414), (8, 23722), (9, 23073), (10, 22097), (11, 20883), (12, 19630), (13, 18444), (14, 17307), (15, 15864), (16, 15062), (17, 13921), (18, 12804), (19, 11822), (20, 11345), (21, 10413), (22, 9596), (23, 9101), (24, 8543), (25, 7749), (26, 7537), (27, 6715), (28, 6450), (29, 5970), (30, 5513), (31, 5191), (32, 4838), (33, 4484), (34, 4199), (35, 4025), (36, 3710), (37, 3514), (38, 3332), (39, 3094), (40, 2869), (41, 2648), (42, 2600), (43, 2319), (44, 2265), (45, 2144), (46, 1991), (47, 1885), (48, 1774), (49, 1694), (50, 1621), (51, 1447), (52, 1339), (53, 1268), (54, 1277), (55, 1137), (56, 1141), (57, 1017), (58, 948), (59, 930), (60, 854), (61, 817), (62, 788), (63, 763), (64, 704), (65, 638), (66, 627), (67, 586), (68, 538), (69, 509), (70, 507), (71, 488), (72, 460), (73, 391), (74, 416), (75, 365), (76, 396), (77, 306), (78, 319), (79, 317), (80, 310), (81, 290), (82, 295), (83, 266), (84, 251), (85, 225), (86, 233), (87, 195), (88, 202), (89, 189), (90, 195), (91, 173), (92, 160), (93, 153), (94, 132), (95, 127), (96, 139), (97, 142), (98, 127), (99, 124), (100, 124), (101, 93), (102, 95), (103, 95), (104, 97), (105, 97), (106, 86), (107, 80), (108, 77), (109, 73), (110, 60), (111, 80), (112, 75), (113, 59), (114, 67), (115, 61), (116, 73), (117, 53), (118, 50), (119, 62), (120, 50), (121, 53), (122, 49), (123, 43), (124, 51), (125, 48), (126, 47), (127, 41), (128, 40), (129, 43), (130, 33), (131, 24), (132, 19), (133, 32), (134, 37), (135, 27), (136, 22), (137, 17), (138, 20), (139, 18), (140, 24), (141, 18), (142, 20), (143, 18), (144, 17), (145, 14), (146, 19), (147, 21), (148, 19), (149, 16), (150, 16), (151, 19), (152, 20), (153, 15), (154, 11), (155, 9), (156, 11), (157, 7), (158, 8), (159, 10), (160, 14), (161, 11), (162, 11), (163, 9), (164, 7), (165, 8), (166, 12), (167, 7), (168, 18), (169, 10), (170, 16), (171, 9), (172, 7), (173, 10), (174, 4), (175, 9), (176, 7), (177, 12), (178, 10), (179, 3), (180, 3), (181, 10), (182, 3), (183, 6), (184, 4), (185, 2), (186, 7), (187, 6), (188, 3), (189, 6), (190, 3), (191, 9), (192, 7), (193, 9), (194, 3), (195, 1), (196, 3), (197, 6), (198, 4), (199, 2), (200, 2), (201, 1), (202, 2), (203, 3), (204, 2), (206, 6), (207, 3), (208, 3), (209, 5), (210, 2), (211, 1), (212, 2), (213, 3), (214, 2), (215, 3), (216, 3), (217, 3), (218, 4), (219, 4), (220, 3), (221, 3), (222, 2), (223, 2), (224, 1), (225, 4), (226, 2), (227, 5), (228, 1), (229, 1), (231, 3), (232, 1), (234, 2), (235, 3), (236, 1), (237, 5), (238, 2), (240, 1), (241, 1), (243, 1), (244, 3), (245, 2), (247, 1), (248, 1), (249, 1), (251, 1), (252, 1), (253, 1), (255, 2), (257, 2), (258, 1), (259, 1), (260, 1), (265, 2), (269, 2), (274, 1), (275, 1), (283, 1), (290, 1), (291, 1), (292, 1), (293, 1), (294, 1), (296, 2), (302, 1), (306, 1), (308, 1), (309, 1), (315, 1), (316, 2), (317, 1), (318, 1), (324, 2), (331, 2), (334, 1), (335, 1), (342, 1), (355, 2), (358, 1), (374, 1), (390, 1), (426, 1), (427, 1), (488, 1), (702, 1)]
# _64 = 0
# _128 = 0
# _256 = 0
# _1 = 0
# _0 = 0
# for length, freq in reversed(d):
#     if length > 0:
#         _0 += freq
#     if length > 1:
#         _1 += freq
#     if length > 64:
#         _64 += freq
#     if length > 128:
#         _128 += freq
#     if length > 256:
#         _256 += freq
#
# print(_0, _1 / _0, _64 / _0, _128 / _0, _256 / _0)

reviews = pd.read_csv('_snappfood.csv')
texts = reviews['commentText'].values
feelings = reviews['feeling'].values

import tensorflow as tf

feelings = tf.constant([0.0 if feeling == 'SAD' else 1.0 for feeling in feelings])
vectors = []
for text in texts:
    vectors.append(sp.encode_as_ids(text)[:64])
    vectors[-1].extend([0] * (64 - len(vectors[-1])))

vectors = tf.constant(vectors)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(8192, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(8192, 64),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(1)
# ])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer=tf.keras.optimizers.Adam(1e-4), metrics=['accuracy'])

print(vectors[0])
print(feelings[0])
print(vectors[0].shape)
history = model.fit(vectors, feelings, epochs=10)
