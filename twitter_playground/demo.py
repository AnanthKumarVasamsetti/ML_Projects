def go_left(matrix,x1,y1,x2,y2,moves,barriers):
    #global moves #+= 1
    global moves
    moves += 1
    if x1 == x2 and y1 == y2:
        return
    temp = y2 - 1
    if temp >= 0 and matrix[x2][temp] == 0:
        y2 = y2 - 1
        go_left(matrix,x1,y1,x2,y2)
    elif temp >= 0 and matrix[x2][temp] != 0:
        global barriers
        barriers += 1
        y2 = y2 - 1
        if matrix[x2][temp] == 1:
            go_down(matrix,x1,y1,x2,y2,moves,barriers)
        elif matrix[x2][temp] == 2:
            go_up(matrix,x1,y1,x2,y2,moves,barriers)

def go_up(matrix,x1,y1,x2,y2,moves,barriers):
    print("in up")
    global moves
    moves += 1
    if x1 == x2 and y1 == y2:
        return moves_arr
    temp = x2 - 1
    if temp >= 0 and matrix[temp][y2] == 0:
        x2 = x2 - 1
        go_up(matrix,x1,y1,x2,y2,moves,barriers)
    elif temp >= 0 and matrix[temp][y2] != 0:
        global barriers
        barriers += 1
        x2 = x2 - 1
        if matrix[temp][y2] == 1:
            go_right(matrix,x1,y1,x2,y2,moves,barriers)
        elif matrix[temp][x2] == 2:
            go_left(matrix,x1,y1,x2,y2,moves,barriers)

def go_right(matrix,x1,y1,x2,y2,moves,barriers):
    print("in right")
    global moves
    moves += 1
    if x1 == x2 and y1 == y2:
        return
    temp = y2 + 1
    if temp < 4 and matrix[x2][temp] == 0:
        y2 = y2 + 1
        go_right(matrix,x1,y1,x2,y2)
    elif temp < 4 and matrix[x2][temp] != 0:
        global barriers
        barriers += 1
        y2 = y2 + 1
        if matrix[x2][temp] == 1:
            go_up(matrix,x1,y1,x2,y2,moves,barriers)
        elif matrix[x2][temp] == 2:
            go_down(matrix,x1,y1,x2,y2,moves,barriers)

def go_down(matrix,x1,y1,x2,y2,moves,barriers):
    print("in down")
    global moves
    moves += 1
    if x1 == x2 and y1 == y2:
        return
    temp = x2 + 1
    if temp < 4 and matrix[temp][y2] == 0:
        x2 = x2 + 1
        go_up(matrix,x1,y1,x2,y2)
    elif temp < 4 and matrix[temp][y2] != 0:
        global barriers
        barriers += 1
        x2 = x2 + 1
        if matrix[temp][y2] == 1:
            go_left(matrix,x1,y1,x2,y2,moves,barriers)
        elif matrix[temp][x2] == 2:
            go_right(matrix,x1,y1,x2,y2,moves,barriers)

def main():
    size = int(input())
    A_pos1, A_pos2 = [int(num) for num in input().strip().split(' ')]
    B_pos1, B_pos2 = [int(num) for num in input().strip().split(' ')]
    matrix = []
    moves = 0
    barriers = 0
    moves_arr = []
    data = []

    for i in range(0,size):
        temp = [int(num) for num in input().strip().split(' ')]
        matrix.append(temp)
    data = []
    data.append(go_left(matrix,A_pos1,A_pos2,B_pos1,B_pos2,moves,barriers))

    data.append(go_right(matrix,A_pos1,A_pos2,B_pos1,B_pos2,moves,barriers))

    data.append(go_up(matrix,A_pos1,A_pos2,B_pos1,B_pos2,moves,barriers))

    data.append(go_down(matrix,A_pos1,A_pos2,B_pos1,B_pos2,moves,barriers))
    data.sort(key = lambda tup: tup[1])
    print(data[0])

if __name__ == "__main__":
    main()
