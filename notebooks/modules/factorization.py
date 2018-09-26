import numpy as np
def factorization(array):
	answer = []
	factorize_number = 0
	input_dict = {}
	for string in array:
		if string not in input_dict.keys():
			input_dict[string] = factorize_number
			factorize_number +=1
	#print(input_dict)

	for string in array:
		answer.append(input_dict.get(string))
	answer = np.asarray(answer)
	return input_dict,answer

# factorization(['sunny',
#  'overcast',
#  'rainy',
#  'sunny',
#  'sunny',
#  'overcast',
#  'rainy',
#  'rainy',
#  'sunny',
#  'rainy',
#  'sunny',
#  'overcast',
#  'overcast',
#  'rainy'])
