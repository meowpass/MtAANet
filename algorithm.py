def findMinArrowShots(points):
    """
    :type points: List[List[int]]
    :rtype: int
    """

    points.sort(key=lambda x: (x[0], x[1]))
    count = 1
    temp = points[0]
    for point in points[1:]:
        if temp[0] == point[0]:
            if temp[1] >= point[1]:
                temp[1] = point[1]
                pass
            pass
        else:
            if temp[1] > point[0]:
                temp[0] = point[0]
                pass
            else:
                count += 1
                temp = point
                pass

    pass
    return count


points = [[9, 12], [1, 10], [4, 11], [8, 12], [3, 9], [6, 9], [6, 7]]
print(findMinArrowShots(points=points))