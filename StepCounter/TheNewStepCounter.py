import matplotlib.pyplot as plt
import math
import statistics

lines = []
path = "C:\\Users\\yusuf\\PycharmProjects\\Data\\step-data\\"
name = "4-100-step-running.csv"
with open(path + name) as reader:
    lines = reader.readlines()


accX = [float(dataline.split(",")[0]) for dataline in lines[4:]]
accY = [float(dataline.split(",")[1]) for dataline in lines[4:]]
# lines[4:] are the rows and the split on "," [1] is the collumn
accZ = [float(dataline.split(",")[2]) for dataline in lines[4:]]

accMag = [math.sqrt(accX[i] ** 2 + accY[i] ** 2 + accZ[i] ** 2) for i in range(len(accZ))]

gyroX = [float(dataline.split(",")[3]) for dataline in lines[4:]]
gyroY = [float(dataline.split(",")[4]) for dataline in lines[4:]]
gyroZ = [float(dataline.split(",")[5]) for dataline in lines[4:]]

gyroMag = [math.sqrt(gyroX[i] * gyroX[i] + gyroY[i] * gyroY[i] + gyroZ[i] * gyroZ[i]) for i in range(len(gyroZ))]


def getStandardDeveation(data):
    Sdev = statistics.stdev(data)
    return Sdev


def findWidth(list):
    for i in range(len(list)):
        if (list[i] > 100):  # tests if running or walking
            return 15
    return 30


def notNoise(list, index):
    width = findWidth(list)
    if (index > len(list) - 25 or index < 25):  # out of bounds check
        return False
    for i in range(index - width, index):
        if (list[i] > list[index]):
            return False
    for i in range(index, index + width):
        if (list[i] > list[index]):
            return False
    return True


def getPeaks(data):
    peaks_X = []
    sum = 0;
    for i in range(len(data)):
        sum = sum + data[i]
    mean = sum / len(data)

    sDev = getStandardDeveation(data)

    for i in range(1, len(data) - 1):
        if (data[i] > data[i - 1] and data[i] > data[i + 1]):
            if (abs(data[i] - mean > 2 * sDev)):
                 if (notNoise(data, i) == True):
                    peaks_X.append(i)
    peaks_Y = [data[index] for index in peaks_X]
    return peaks_X, peaks_Y


plt.plot(accMag[:], 'r-')
peaksX, peaksY = getPeaks(accMag[:])

plt.plot(peaksX, peaksY, "b.")
print(f"Total steps are : {len(peaksX) * 2}")
plt.show()
