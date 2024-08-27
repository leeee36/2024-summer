import torch
import numpy as np

# 텐서(tensor)는 배열(array)이나 행렬(matrix)과 매우 유사한 특수한 자료구조

print("---------------------------------------------------------------------")

#텐서 초기화

#데이터로부터 직접 생성하기
data = [[1,2], [3,4]]
x_data = torch.tensor(data)

#넘파이로 생성하기
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

#다른 텐서로부터 텐서 생성하기
x_ones = torch.ones_like(x_data) #x_data의 속성을 유지함
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) #x_data의 속성을 덮어쓰기함
print(f"Random Tensor: \n {x_rand} \n")

print("---------------------------------------------------------------------")

#무작위 or 상수값 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zero_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zero Tensor: \n {zero_tensor}")

print("---------------------------------------------------------------------")

#텐서의 속성
tensor = torch.rand(3,4)
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

print("---------------------------------------------------------------------")

#텐서 연산
#기본적으로 텐서는 cpu에 생성된다. 
#.to 메소드를 사용하면 gpu의 가용성을 확인한 후 gpu로 텐서를 명시적으로 이동할 수 있다.

if torch.cuda.is_available():
    tensor = tensor.to("cuda")

tensor = torch.ones(4,4)
print(f"first row: {tensor[0]}")
print(f"first column: {tensor[:,0]}")
print(f"last column: {tensor[..., -1]}")
tensor[:,1] = 0
print(tensor)

t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)

print("---------------------------------------------------------------------")

#산술 연산
#두 텐서 간의 행렬곱 연산
#y1, y2, y3는 모두 같은 값을 가짐
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


#요소별 곱 계산
z1 = tensor * tensor 
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

print("---------------------------------------------------------------------")

#단일요소 텐서
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

#바꿔치기 연산 
print(f"{tensor} \n")
tensor.add_(5)
print(tensor)

#바꿔치기 연산은 메모리를 일부 절약하지만, 기록이 즉시 삭제되어 도함수 계산에 문제가 발생할 수 있다. 
#따라서 사용을 권장하지 않음.

print("---------------------------------------------------------------------")

#numpy 변환
#텐서를 numpy로 변환하기
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

print("---------------------------------------------------------------------")

#numpy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
