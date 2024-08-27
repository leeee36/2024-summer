"""
torch.autograd 를 사용한 자동 미분

신경망을 학습할 때 가장 자주 사용되는 알고리즘은 "역전파"이다.
이 알고리즘에서 매개변수(모델 가중치)는 
주어진 매개변수에 대한 손실함수의 변화(gradient)에 따라 조정됨

torch.autograd는 자동 미분엔진으로 모든 계산 그래프에 대한 변화도의 자동계산 지원
"""

import torch
x = torch.ones(5)
y = torch.zeros(3)
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# 이 신경망에서 w, b는 최적화를 해야하는 매개변수이며
# 위 변수들에 대한 손실함수의 변화도를 계산할 수 있어야 함.

"""
연산 그래프를 구성하기 위해 텐서에 적용하는 함수는 Function 클래스의 객체이다.
이 객체는 순전파 방향으로 함수를 계산하는 방법과
역방향 전파 단계에서 도함수를 계산하는 방법을 알고 있다.
"""

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")


# 변화도(Gradient) 계산하기
loss.backward()
print(w.grad)
print(b.grad)

# 변화도 추적 멈추기
"""
기본적으로 requires_grad=True 인 모든 텐서들은 연산 기록을 추적하고 변화도 계산 지원함.
그러나 모델을 학습한 뒤 입력 데이터를 단순히 적용하는 경우처럼 순전파 연산만 필요하다면
위의 추적이 필요없게 됨.
"""
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

# 다른 방법
z = torch.matmul(x, w)+b
z_det = z.detach()
print(z_det.requires_grad)

"""
변화도 추적을 멈추는 이유:
1. 신경망의 일부 매개변수를 고정된 매개변수로 표시
2. 변화도를 추적하지 않는 텐서의 연산이 더 효율적임. 따라서 연산 속도 향상됨
"""

#-------------------------------------------------------------
# 참고 #

"""
순전파 단계에서, autograd는 다음 두 가지 작업을 동시에 수행합니다:

요청된 연산을 수행하여 결과 텐서를 계산하고,

DAG에 연산의 변화도 기능(gradient function) 를 유지(maintain)합니다.

역전파 단계는 DAG 뿌리(root)에서 .backward() 가 호출될 때 시작됩니다. autograd는 이 때:

각 .grad_fn 으로부터 변화도를 계산하고,

각 텐서의 .grad 속성에 계산 결과를 쌓고(accumulate),

연쇄 법칙을 사용하여, 모든 잎(leaf) 텐서들까지 전파(propagate)합니다.
"""

"""
PyTorch에서 DAG들은 동적(dynamic)입니다. 주목해야 할 중요한 점은 그래프가 처음부터(from scratch) 다시 생성된다는 것입니다; 매번 .bachward() 가 호출되고 나면, autograd는 새로운 그래프를 채우기(populate) 시작합니다. 이러한 점 덕분에 모델에서 흐름 제어(control flow) 구문들을 사용할 수 있게 되는 것입니다; 매번 반복(iteration)할 때마다 필요하면 모양(shape)이나 크기(size), 연산(operation)을 바꿀 수 있습니다.
"""

#-------------------------------------------------------------
print("#-------------------------------------------------------------")
#-------------------------------------------------------------

# 선택적으로 읽기(optional reading)
# 텐서 변화도와 야코비안 곱(jacobian matrix)
"""
대부분의 경우, 스칼라 손실 함수를 가지고 일부 매개변수와 관련한 변화도를 계산해야 함
그러나 출력 함수가 임의의 텐서인 경우, pytorch는 실제 변화도가 아닌 야코비안 곱을 계산함
"""

inp = torch.eye(4, 5, requires_grad=True)
out = (inp+1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n {inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n {inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n {inp.grad}")

# 제대로 된 변화도를 계산하기 위해서는 grad 속성을 먼저 0으로 만들어야 함
# 실제 학습 과정에서는 옵티마이저가 이 과정을 도와줌