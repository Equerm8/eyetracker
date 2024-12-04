def find_max_arr_val(cols, rows, searches, counter) -> tuple:
    maxVal = -1
    found = False
    foundIndex1 = 0; foundIndex2 = 0
    
    for i in range(rows-1, -1, -1):
        for j in range(cols-1, -1, -1):
            if searches[i][j] == False and counter[i][j] > maxVal:
                
                if found == True:
                    searches[foundIndex1][foundIndex2] = False
                    maxVal = counter[i][j]
                    searches[i][j] = True
                    foundIndex1 = i
                    foundIndex2 = j

                else:
                    maxVal = counter[i][j]
                    searches[i][j] = True
                    foundIndex1 = i
                    foundIndex2 = j
                    found = True

    return foundIndex1, foundIndex2

def is_end(rows, cols, searches):
    for i in range(rows):
        for j in range(cols):
            if searches[i][j] == False:
                return False
    return True
    
# def add_freq(imgs, i, j, xGaze, yGaze, counter):
#     xBorder = 0
#     yBorder = 0
#     xBorderLeft = 0
#     yBorderLeft = 0
#     iY = 0
#     exitLoop = False
#     for iHeight in range(i+1):
#         for jWidth in range(j+1):
#             xBorder += imgs[iHeight][jWidth].shape[1]
#             if jWidth != 0:
#                 xBorderLeft += imgs[iHeight][jWidth-1].shape[1]
#             if iY <= iHeight:
#                 yBorder += imgs[iY][jWidth].shape[0]
#                 if iHeight != 0:
#                     yBorderLeft += imgs[iHeight-1][jWidth].shape[0]
#                 iY += 1
#             if xGaze <= xBorder and yGaze <= yBorder and xGaze >= xBorderLeft and yGaze >= yBorderLeft:
#                 counter[iHeight][jWidth] += 1
#                 exitLoop = True
#                 return exitLoop
#         xBorder = 0
#     return exitLoop

def add_freq(imgs, i, j, xGaze, yGaze, counter, rows, cols):
    height, width = imgs[i][j].shape
    window_height = height // rows
    window_width = width // cols
    examined_height = window_height * i
    examined_width = window_width * j
    prev_height = examined_height - window_width
    prev_width = examined_width - window_width 

    if xGaze <= examined_width and yGaze <= examined_height and xGaze >= prev_width and yGaze >= prev_height:
        counter[i][j] += 1
        return True
    else:
        return False
