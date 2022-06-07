import time
import random 
import numpy as np
import torch 
from torch.autograd import Variable

np.random.seed(42)
torch.manual_seed(42)

patience = 100
use_cuda=True

device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

orig_scores = None

def attack_untargeted(model, train_dataset, x0, y0, alpha = 0.2, beta = 0.001, iterations = 1000):
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

    num_samples = 1000
    best_theta, g_theta = None, float('inf')
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
            initial_lbd = torch.norm(theta)
            theta = theta/torch.norm(theta)
            lbd, count = fine_grained_binary_search(model, x0, y0, theta, initial_lbd, g_theta)
            query_count += count
            if lbd < g_theta:
                best_theta, g_theta = theta, lbd
                print("--------> Found distortion %.4f" % g_theta)

    timeend = time.time()
    print("==========> Found best distortion %.4f in %.4f seconds using %d queries" % (g_theta, timeend-timestart, query_count))

    
    
    timestart = time.time()
    g1 = 1.0
    theta, g2 = best_theta.clone(), g_theta
    torch.manual_seed(0)
    opt_count = 0
    stopping = 0.01
    prev_obj = 100000
    for i in range(iterations):
        gradient = torch.zeros(theta.size()).to(device)
        q = 10
        min_g1 = float('inf')
        for _ in range(q):
            u = torch.randn(theta.size()).type(torch.FloatTensor).to(device)
            u = u/torch.norm(u)
            ttt = theta+beta * u
            ttt = ttt/torch.norm(ttt)
            g1, count = fine_grained_binary_search_local(model, x0, y0, ttt, initial_lbd = g2, tol=beta/500)
            opt_count += count
            gradient += (g1-g2)/beta * u
            if g1 < min_g1:
                min_g1 = g1
                min_ttt = ttt
        gradient = 1.0/q * gradient

        if (i+1)%50 == 0:
            print("Iteration %3d: g(theta + beta*u) = %.4f g(theta) = %.4f distortion %.4f num_queries %d" % (i+1, g1, g2, torch.norm(g2*theta), opt_count))
            if g2 > prev_obj-stopping:
                break
            prev_obj = g2

        min_theta = theta
        min_g2 = g2
    
        for _ in range(15):
            new_theta = theta - alpha * gradient
            new_theta = new_theta/torch.norm(new_theta)
            new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
            opt_count += count
            alpha = alpha * 2
            if new_g2 < min_g2:
                min_theta = new_theta 
                min_g2 = new_g2
            else:
                break

        if min_g2 >= g2:
            for _ in range(15):
                alpha = alpha * 0.25
                new_theta = theta - alpha * gradient
                new_theta = new_theta/torch.norm(new_theta)
                new_g2, count = fine_grained_binary_search_local(model, x0, y0, new_theta, initial_lbd = min_g2, tol=beta/500)
                opt_count += count
                if new_g2 < g2:
                    min_theta = new_theta 
                    min_g2 = new_g2
                    break

        if min_g2 <= min_g1:
            theta, g2 = min_theta, min_g2
        else:
            theta, g2 = min_ttt, min_g1

        if g2 < g_theta:
            best_theta, g_theta = theta.clone(), g2
        
        #print(alpha)
        if alpha < 1e-4:
            alpha = 1.0
            print("Warning: not moving, g2 %lf gtheta %lf" % (g2, g_theta))
            beta = beta * 0.1
            if (beta < 0.0005):
                break

    target = predict(model, x0 + g_theta*best_theta)
    timeend = time.time()
    print("\nAdversarial Example Found Successfully: distortion %.4f target %d queries %d \nTime: %.4f seconds" % (g_theta, target, query_count + opt_count, timeend-timestart))
    return x0 + g_theta*best_theta, g_theta*best_theta

def fine_grained_binary_search_local(model, x0, y0, theta, initial_lbd = 1.0, tol=1e-5):
    nquery = 0
    lbd = initial_lbd
     
    if predict(model, x0+lbd*theta) == y0:
        lbd_lo = lbd
        lbd_hi = lbd*1.01
        nquery += 1
        while predict(model, x0+lbd_hi*theta) == y0:
            lbd_hi = lbd_hi*1.01
            nquery += 1
            if lbd_hi > 20:
                return float('inf'), nquery
            if nquery > patience:
                print("Crossed patience, breaking execution")
                return float('inf'), nquery
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

    while (lbd_hi - lbd_lo) > tol:
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

def fine_grained_binary_search(model, x0, y0, theta, initial_lbd, current_best):
    nquery = 0
    if initial_lbd > current_best: 
        if predict(model, x0+current_best*theta) == y0:
            nquery += 1
            return float('inf'), nquery
        lbd = current_best
    else:
        lbd = initial_lbd

    lbd_hi = lbd
    lbd_lo = 0.0

    while (lbd_hi - lbd_lo) > 1e-5:
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

def opt_attack(model, dataloader, num_samples=1):

    global orig_scores

    model = model.to(device)
    model.eval()

    attack_done = False

    print("Running on first {} images ".format(num_samples))

    for i, (image, label) in enumerate(dataloader):
        image, label = image.to(device), label.to(device)

        print("======== Image %d =========" % i)
        print("Original label: ", label)

        adversarial, pertub = attack_untargeted(model, dataloader, image, label, alpha=10, beta=0.005, iterations = 100)

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

    #attack_mnist(alpha=2, beta=0.005, isTarget= False)
    #attack_cifar10(alpha=5, beta=0.001, isTarget= False)
    #attack_imagenet(arch='resnet50', alpha=10, beta=0.005, isTarget= False)