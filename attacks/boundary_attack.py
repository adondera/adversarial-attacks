import time
import random 
import numpy as np
import torch 
from torch.autograd import Variable
from zmq import device

alpha = 0.2
beta = 0.001
patience = 100
use_cuda=True

device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

orig_scores = None

def attack_untargeted(model, train_dataset, x0, y0, alpha = 0.2, beta = 0.001):
    """ Attack the original image and return adversarial example
        model: (pytorch model)
        train_dataset: set of training data
        (x0, y0): original image
    """

    global orig_scores

    predicted, orig_scores = predict(model, x0, return_score=True)
    print("Predicted label: ", predicted)

    if (predicted != y0):
        print("Fail to classify the image. No need to attack.")
        return x0, None

    num_samples = 10
    best_theta = None
    best_distortion = float('inf')
    g_theta = None
    query_count = 0

    print("Searching for the initial direction on %d samples: " % (num_samples))
    timestart = time.time()
    samples = set(random.sample(range(len(train_dataset)), num_samples))
    for i, (xi, yi) in enumerate(train_dataset):
        if i not in samples:
            continue
        query_count += 1
        if predict(model, xi) != y0:
            theta = xi - x0
            #query_count += query_search_each
            lbd, count = fine_grained_binary_search(model, x0, y0, theta)
            query_count += count
            distortion = torch.norm(lbd*theta)
            if distortion < best_distortion:
                best_theta, g_theta = theta, lbd
                best_distortion = distortion
                print("--------> Found distortion %.4f and g_theta = %.4f" % (best_distortion, g_theta))

    timeend = time.time()
    print("==========> Found best distortion %.4f and g_theta = %.4f in %.4f seconds using %d queries" % (best_distortion, g_theta, timeend-timestart, query_count))

    #query_limit -= query_count

  
    timestart = time.time()

    #query_search_each = 200  # limit for each lambda search
    #iterations = (query_limit - query_search_each)//(2*query_search_each)
    iterations = 1000
    g1 = 1.0
    g2 = g_theta
    theta = best_theta
    now_o = g2*theta
    delta = 0.01
    epsilon = 0.001

    success_count = 0
    for i in range(iterations):
        u = torch.randn(theta.size()).type(torch.FloatTensor).to(device)
        new_o = now_o + u*delta
        new_o = new_o *( torch.norm(now_o) / torch.norm(new_o))
        if predict(model, new_o) != y0:
            success_count += 1
        new_o = new_o - epsilon*(new_o)/torch.norm(new_o)
        if predict(model, x0+new_o) != y0:
            now_o = new_o
        if (i+1)%10 == 0:
            print("Iteration %3d distortion %.4f query %d" % (i+1, torch.norm(now_o), query_count+(i+1)*2))

    distortion = torch.norm(now_o)
    target = predict(model, now_o)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (distortion, target, query_count + iterations, timeend-timestart))
    return x0+now_o, now_o

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
   
    if predict(model, x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while predict(model, x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if nquery > patience:
                print("Crossed patience, breaking execution")
                break
    else:
        lbd_hi = lbd
        lbd_lo = lbd*0.99
        nquery += 1
        while predict(model, x0+lbd_lo*theta) != y0 :
            lbd_lo = lbd_lo*0.99
            nquery += 1
            if nquery > patience:
                print("Crossed patience, breaking execution")
                break

    while (lbd_hi - lbd_lo) > 1e-8:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict(model, x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
    return lbd_hi, nquery

def fine_grained_binary_search(model, x0, y0, theta, initial_lbd = 1.0):
    nquery = 0
    lbd = initial_lbd
    while predict(model, x0 + lbd*theta) == y0:
        lbd *= 2.0
        nquery += 1
        if nquery > patience:
            print("Crossed patience, breaking execution")
            break

    num_intervals = 100

    lambdas = np.linspace(0.0, lbd, num_intervals)[1:]
    lbd_hi = lbd
    lbd_hi_index = 0
    for i, lbd in enumerate(lambdas):
        nquery += 1
        if predict(model, x0 + lbd*theta) != y0:
            lbd_hi = lbd
            lbd_hi_index = i
            break

    lbd_lo = lambdas[lbd_hi_index - 1]

    while (lbd_hi - lbd_lo) > 1e-7:
        lbd_mid = (lbd_lo + lbd_hi)/2.0
        nquery += 1
        if predict(model, x0 + lbd_mid*theta) != y0:
            lbd_hi = lbd_mid
        else:
            lbd_lo = lbd_mid
        
        if nquery > patience:
            print("Crossed patience, breaking execution")
            break

    return lbd_hi, nquery

def predict(model, image, return_score=False):
    image = torch.clamp(image,0,1)
    image = Variable(image)
    if torch.cuda.is_available():
        image = image.cuda()
    with torch.no_grad():
        output = model(image)
        _, predict = torch.max(output.data, 1)

    if return_score:
        return predict, output
    else:
        return predict

def boundary_attack(model, dataloader, num_samples=1):

    model = model.to(device)
    model.eval()

    print("Running on first {} images ".format(num_samples))

    attack_done = False

    for i, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)

        print("======== Image %d =========" % i)
        print("Original label: ", label)

        adversarial, pertub = attack_untargeted(model, dataloader, image, label, alpha = alpha, beta = beta)

        if pertub is not None:
            predicted, adv_scores = predict(model, adversarial, return_score=True)
            print("Predicted label for adversarial example: ", predicted)
            print("Original label: ", label)

            attack_done = True

        if (i+1) >= num_samples and attack_done:
            print("Breaking execution")
            break
    
    return adversarial, pertub, image, label, orig_scores, adv_scores

if __name__ == '__main__':
    pass
