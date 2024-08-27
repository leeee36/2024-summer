import random
import matplotlib.pyplot as plt

door = ["a", "b", "c"]

# ===== 정답을 바꾸지 않는 경우 ===== #
def AnsWithNoChange(num):
    point = 0

    for _ in range(num):
        choice = random.choice(door)
        ans = random.choice(door)
        # print(f'choice : {choice} & ans : {ans}')
        if choice == ans:
            point += 1
    
    return point

# ===== 정답을 바꾸는 경우 ===== #
def AnsWithChange(num):
    point = 0

    for _ in range(num):
        choice = random.choice(door)
        ans = random.choice(door)

        if choice == ans :
            # 이 경우엔 다른 문을 열어서 보여준 후 선택을 바꾼다한들 무조건 틀리는 상황
            # 따라서 패스함
            pass
        elif choice != ans :
            # 이 경우엔 정답이 아닌 문을 내가 골랐고 나머지 정답이 아닌 문을 열어서 보여주는 경우이므로
            # 내 선택과 열어서 보여준 문이 아닌 문은 무조건 정답인 경우임
            # 따라서 choice 와 ans 가 다르다면 무조건 포인트를 얻게 됨
            point += 1
    
    return point

if __name__ == "__main__":

    """
    # a = AnsWithNoChange(num)
    # b = AnsWithChange(num)

    # print(f'정답을 바꾸지 않는 경우: {a} / {num}')
    # print(f'정답을 바꾸는 경우 : {b} / {num}')
    """

    num = 100
    no_change = []
    change = []

    for _ in range(num):
        # 100번의 경우에 대한 결과를 본 다음
        # 그 결과를 100번 실행했을 때 평균을 확인
        no_change.append(AnsWithNoChange(num))
        change.append(AnsWithChange(num))

    mean_of_nochange = sum(no_change) / num
    mean_of_change = sum(change) / num

    print("=======================================")
    print("정답을 바꾸지 않는 경우 평균 : {:2.4f}".format(mean_of_nochange))
    print("정답을 바꾸는 경우 평균 : {:2.4f}".format(mean_of_change))
    print("=======================================")

    plt.hist(no_change, label='no_change')
    plt.hist(change, label='change')
    plt.legend()
    plt.xlabel('corrects num of one execution') # 각 경우에 대해 1회 실행 시 나온 정답 수
    plt.ylabel('num of corrects') # 정답을 맞춘 갯수
    plt.show()
