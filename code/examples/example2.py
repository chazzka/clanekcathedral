from mock.randomdatagenerator import *
import sys
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest

from articletester.articletester import *

if __name__ == "__main__":


    testData = [
        (68.05419491643804, 89.91148343538444),
        (129.48248275891228, 109.97046948747158),
        (108.86887033962708, 103.57415497384716),
        (99.80580341510776, 97.95225289939329),
        (153.65614322995657, 99.82911043838284),
        (83.70928858710899, 82.96123945060532),
        (87.12237123839586, 104.7427080513091),
        (154.7510181743543, 101.33923076624961),
        (82.29798878492062, 99.03708591654502),
        (118.75491854637833, 101.32097466053382),
        (75.35082291816147, 102.48679942192068),
        (141.0284217748827, 107.49550769404989),
        (71.39484320367505, 99.00766742395803),
        (168.82863024460949, 106.78845333435365),
        (135.257246619369, 99.47650353853649),
        (112.89898980748973, 92.75601920569623),
        (88.88245113026198, 104.42925689253585),
        (103.91756003896342, 93.04362610502862),
        (71.99886694171151, 113.02548727845216),
        (110.04083176494814, 104.44844462419853),
        (89.61355190610622, 104.41369639480493),
        (124.7250484022647, 99.04147424907568),
        (161.31158717074825, 99.62380247613424),
        (121.24407412382922, 98.90379119322402),
        (111.84290437658092, 108.40267562822892),
        (83.66049782578969, 106.65500800666227),
        (71.28948421494854, 101.83814769790217),
        (143.53000490506184, 106.434865897127),
        (113.36135976002086, 98.99314596229354),
        (98.05914788278959, 104.89834218022678),
        (99.55156851945534, 93.80357188782355),
        (152.7523179097504, 100.69038307439182),
        (166.381473207271, 106.28771325842726),
        (64.18895612934139, 99.04723163542849),
        (84.45563170507941, 98.82721377540241),
        (148.79900026541378, 97.48722522041834),
        (91.9980207118104, 94.21785625582402),
        (99.07107096687935, 106.94002839548172),
        (81.1723947781936, 105.65676115850503),
        (74.75636935128993, 101.48091010580737),
        (154.27017653860474, 112.7167374667188),
        (136.40853506440732, 93.86791778526266),
        (71.81968228758149, 100.46908150845573),
        (156.13042320677377, 98.2937238696687),
        (111.15714125524568, 100.51676452526286),
        (62.12129695493343, 97.704047344266),
        (164.2790540716582, 104.02427761437303),
        (139.7181795566089, 100.70280659511641),
        (146.6963867554323, 107.40327369013318),
        (162.3160441580509, 98.69840995257472),
        (74.29347418346147, 95.06817066425914),
        (106.95753566661342, 98.38996263735824),
        (70.53595864474869, 96.90801886766411),
        (140.89537131449063, 94.12456030505187),
        (84.47863810657438, 105.98470225687849),
        (74.36885140358905, 103.87940578735186),
        (90.37428374051852, 101.06059876419354),
        (104.8714318076571, 100.37128557403564),
        (68.9463654088147, 98.08409388098194),
        (66.38601264834524, 100.4456269959929),
        (121.78088985819826, 96.89152838866835),
        (158.70737852195134, 95.84947242691199),
        (63.847162350075465, 111.15761107563169),
        (60.65162269863965, 94.30644909275156),
        (163.03923195108553, 97.02194640846122),
        (154.05884510061404, 108.48142574213722),
        (153.44568754885154, 107.69620917716482),
        (130.02418754345285, 94.7134225127952),
        (149.9693851054801, 109.72510063731633),
        (61.76212409930334, 111.33568501173139),
        (165.56981860468045, 96.39644497351865),
        (107.3327501750198, 108.74707028471491),
        (147.7551629140039, 111.31357145335875),
        (69.15768929260861, 96.09560592944054),
        (111.90247757354112, 93.45828139620035),
        (92.13305880541684, 102.29694285838795),
        (116.60925768105943, 99.61146742403535),
        (128.84571332368733, 108.39079307399493),
        (60.4590098533555, 99.43577990241143),
        (122.0359969689888, 110.1466169498697),
        (151.28931803205677, 104.08852895589543),
        (96.00764863133324, 105.67547263697634),
        (135.17720484962052, 103.13232436774291),
        (62.51709366854846, 99.44769909757457),
        (114.37954900983584, 102.31579664636709),
        (132.51888478501036, 95.20431965366525),
        (130.29442554453843, 81.41148976493042),
        (166.76190683515512, 102.7782415311142),
        (151.4050232484759, 105.04083050346868),
        (153.25403303090718, 101.1192035972846),
        (71.73945189385664, 101.45390803569224),
        (77.3510797145766, 94.91327640412688),
        (138.01349750283816, 104.9881868867989),
        (97.73661316823932, 95.91001582320486),
        (156.93643628485702, 93.89835961593735),
        (98.60773175484354, 101.73786652586732),
        (109.15196766216508, 99.07576156276708),
        (75.92083169416915, 89.92828522535822),
        (149.49916565464048, 106.56783884357486),
        (120.85270460209145, 98.8633579267664),
        (36.2181988109781, 101.2635876953761),
        (30.159210186593317, 96.5723128303734),
        (5.994498921157526, 96.33482267457644),
        (20.56950606232198, 108.35925959264134),
        (33.10693673471788, 92.7999611591106),
        (9.424898275608895, 100.89113857942685),
        (38.05657600077847, 101.8612353443965),
        (31.30475557557159, 104.10645814600385),
        (39.66380532599539, 107.0785942746284),
        (20.17140982449247, 97.71011388168678),
        (48.784066326452056, 97.28686363869538),
        (44.94638308671609, 100.16881151442577),
        (3.280371943070021, 101.5319942424128),
        (23.583423204594272, 106.13106540822986),
        (43.086082072682345, 91.88517956045918),
        (44.859068693277216, 103.71803331060514),
        (31.58075026958992, 100.42591202072288),
        (33.43869726849375, 89.26314953402378),
        (3.9365353012314808, 111.15358719928788),
        (6.087136198268422, 111.12076614361804),
        (48.861112446636135, 104.74460015436347),
        (4.852234531267941, 105.10599052778366),
        (32.87742391181237, 94.84255859756401),
        (39.69535325354088, 97.66658106038224),
        (9.10820854120753, 94.675866237378),
        (45.48865403478942, 95.90306047811437),
        (38.70525685306013, 93.68228985504823),
        (48.53404303992201, 94.45528257929105),
        (47.21260709814124, 90.3980656064545),
        (28.444731644624106, 91.92101149175494),
        (45.4719725990716, 101.22231853648356),
        (13.01843081116037, 97.40857157542604),
        (6.104372468377578, 99.62507404093391),
        (49.42171557358341, 94.50365349256232),
        (15.617983966232185, 97.27421239501435),
        (23.82525498723705, 103.54424236777531),
        (16.721566659799986, 104.2993606767944),
        (33.60953740997598, 105.18189496886023),
        (40.146448330214604, 98.40156622371957),
        (6.79794129490256, 93.52782956963894),
        (38.70635505511667, 83.65914706642522),
        (10.082303598409798, 105.33399950983778),
        (12.32718023613546, 96.95366504770072),
        (18.364329103546428, 99.79567661899375),
        (15.17452279762448, 100.70839986609342),
        (7.951372342108881, 106.27436179017523),
        (46.482159212344385, 95.99538265654552),
        (11.098929245596784, 98.42156598923263),
        (30.015501909386682, 106.73057717236682),
        (34.788543709410824, 103.33580914943451),
        (9.09914287783774, 100.11369122193558),
        (29.077936627669697, 97.69052072620694),
        (11.516573341102937, 103.51091171369686),
        (20.717617042375196, 95.90808805448518),
        (17.03919499558493, 103.54068364670134),
        (14.754639496737381, 93.45313782979527),
        (5.149060883815726, 98.27829146168882),
        (41.11991997539704, 92.55598441253416),
        (4.080025181786273, 104.11878806085394),
        (47.6642423013127, 100.74478445732214),
        (21.879970228894162, 100.62965938180453),
        (45.077551991237534, 88.18408620525494),
        (4.875152004248245, 99.76160482528736),
        (44.30017224721028, 94.08514265945544),
        (12.334435837818708, 99.5117373699343),
        (7.208856109898648, 92.73959078559916),
        (24.193426840424117, 94.83709795193404),
        (5.785172998256294, 103.55564037374029),
        (5.987638123368444, 90.14090436894514),
        (6.74698437689144, 104.48236171387687),
        (43.36196600364147, 97.67972420104107),
        (37.30045933925892, 107.57477462234836),
        (40.76986066900925, 102.5042804851215),
        (25.068866486671727, 91.09494199264486),
        (45.44313395538027, 109.35523688096124),
        (34.83566031752047, 89.22531943375357),
        (43.72614067042335, 100.51578658091186),
        (22.48704983654412, 97.45936508911025),
        (41.97971324552812, 110.31547793583879),
        (7.383535229645489, 97.62072323011974),
        (13.198237187719213, 106.56379520661483),
        (36.29456740767695, 101.6893024670068),
        (43.299693649550555, 97.6955915449442),
        (48.87462254056483, 98.11154815691634),
        (24.9375061089271, 98.1882244889977),
        (2.523705018964395, 96.46946527558454),
        (12.849797406152035, 111.0499264377028),
        (3.9428553858352533, 104.1303319012613),
        (10.946962401329435, 108.97147688253304),
        (44.66079970501927, 103.35870487185176),
        (27.11835377432829, 98.72750142914424),
        (36.304933753414936, 110.84494374353729),
        (40.41591109167092, 96.74435078758898),
        (41.46513056432307, 100.47112993894036),
        (36.67455030642309, 99.0880863003455),
        (9.813751641913647, 100.2248615875429),
        (9.999367580972063, 104.5016281349915),
        (37.10124700084062, 96.02096300939077),
        (8.446420006462569, 93.83346690921437),
        (11.410957263578176, 105.3522121536282),
    ]

    testData = np.asarray(testData)

    predictedList = IsolationForest(random_state=0, contamination=0.01, max_samples='auto').fit_predict(testData)

    regular = testData[predictedList == 1]

    outlier = testData[predictedList == -1]

    fi, ax = plotXYDataTwoClasses(regular, outlier, plt.subplots(), c=[
        'black', 'white'], labels=["Regular", "Outlier"], savepath=f"figures/{Path(__file__).stem}.svg")

    plt.show()

    print("done")
    sys.exit(0)