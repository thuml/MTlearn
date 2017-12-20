import torch
import numpy as np

epsilon = 0.00001

def TensorUnfold(input_tensor, k):
    shape_tensor = list(input_tensor.size())
    num_dim = len(shape_tensor)
    permute_order = [k] + np.delete(range(num_dim), k).tolist()
    middle_result = input_tensor.permute(*permute_order)
    shape_middle = list(middle_result.size())
    result = middle_result.view([shape_middle[0], np.prod(shape_middle[1:])])
    return result

def TensorProduct(tensor1, tensor2, axes=(0, 0)):
    shape1 = list(tensor1.size())
    shape2 = list(tensor2.size())
    shape_out = np.delete(shape1, axes[0]).tolist() + np.delete(shape2, axes[1]).tolist()
    result = torch.matmul(torch.t(TensorUnfold(tensor1, axes[0])), TensorUnfold(tensor2, axes[1]))
    return result.resize_(shape_out)

def UpdateCov(weight_matrix, tensor1, tensor2):
    size0 = weight_matrix.size(0)
    final_result = torch.mm(weight_matrix.view(size0, -1), torch.t(torch.matmul(tensor1, torch.matmul(weight_matrix, torch.t(tensor2))).view(size0, -1)))
    return final_result + epsilon * torch.eye(final_result.size(0)).cuda()

def MultiTaskLoss(weight_matrix, tensor1, tensor2, tensor3):
    size_dim0 = weight_matrix.size(0)
    size_dim1 = weight_matrix.size(1)
    size_dim2 = weight_matrix.size(2)
    middle_result1 = torch.matmul(weight_matrix, torch.t(tensor3))
    middle_result2 = torch.matmul(tensor2, middle_result1)
    final_result = torch.matmul(tensor1, middle_result2.permute(1,0,2)).permute(1,0,2).contiguous()
    return torch.mm(weight_matrix.view(1, -1), final_result.view(-1, 1)).view(1)
    
if __name__ == "__main__":
    tensor1 = torch.randn(2,3,4)
    tensor2 = torch.randn(3,4,5)
    result = TensorProduct(tensor1, tensor2, (1, 0))
    print(tensor1)
    print(tensor2)
    print(result)
