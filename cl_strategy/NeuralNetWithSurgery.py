# https://paperswithcode.com/paper/neural-network-surgery-with-sets

# Idea is that if task 0 is trained on model 0, then model 0's outputs only response to the class

# Example below, only indices 1 and 4 are activated, elsewhere are 0, so trace back parameters of what correspond to
# indices 1 and 4, freeze them, then train on other task.
"""
net_0 prediction
                [5.7552214e-04 1.3564850e+01 6.8130658e-04 6.1788253e-04 1.8430006e+01
                 2.8869390e-04 6.9698814e-04 5.0782855e-04 5.5186177e-04 1.2267524e-03]
"""


# Another idea is that encode images, then decode to get back to the image, compare if decoded image == input image,
# then correct, if not, then unknown

