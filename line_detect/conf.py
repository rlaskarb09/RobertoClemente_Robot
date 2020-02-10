## Picture settings
# initial grayscale threshold
threshold = 120
# max grayscale threshold
threshold_max = 180
#min grayscale threshold
threshold_min = 40
# iterations to find balanced threshold
th_iterations = 10
# min % of white in roi
white_min=3
# max % of white in roirow = len(arr)
#     pri_dig = 0
#     sed_dig = 0
#     for i in range(row):
#         pri_dig += arr[i][i]
#         sed_dig += arr[i][row-i]
#     return abs(pri_dig - sed_dig)
white_max=12

#Driving setting
rightCenter = -5
rightAngle  = 90

#PID setting
<<<<<<< Updated upstream
kp =1
kd = 0
=======
kp = 0.66
kd = 0.33
>>>>>>> Stashed changes
# angle error
angle_step = 0.2
# shift error
shift_step = 0.8
