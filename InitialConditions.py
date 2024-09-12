# File Details

import numpy as np



## ------SMOOTH Functions ------
NUM_OF_SMOOTH_FUNCTIONS = 2

def smooth(params, ic='any'): 
    if ic == 'any':
        id = np.random.randint(0, high=NUM_OF_SMOOTH_FUNCTIONS)
    
        if id == 0: 
            output_function = lambda x: sine(x, params)
        else: 
            output_function = lambda x:  constant(x, params)
    
    elif ic == 'sine':
        output_function = lambda x: sine(x, params)
    
    elif ic == 'constant':
        output_function = lambda x: constant(x, params)

    return output_function 

def sine(input, params): 
    bias = params['bias']
    freq = params['freq']
    output = np.sin(freq*np.pi * input) + bias * np.ones((len(input)))
    return output

def constant(input, params): 
    bias = params['bias']
    output = bias * np.ones((len(input)))
    return output


## ------NON-SMOOTH Functions ------
NUM_OF_NONSMOOTH_FUNCTIONS = 6

def nonSmooth(params, ic='any', domain=[-5, 5]): 
    if ic == 'any':
        id = np.random.randint(0, high=NUM_OF_NONSMOOTH_FUNCTIONS)
        
        if id == 0: 
            output_function = lambda x: monomial(x, params)
        elif id == 1: 
            output_function = lambda x: topHat(x, params, domain=[domain[0], domain[1]])
        elif id == 2: 
            output_function = lambda x: unitPulse(x, params, domain=[domain[0], domain[1]])
        elif id == 3: 
            output_function = lambda x: sawtooth(x, params, domain=[domain[0], domain[1]])
        elif id == 4: 
            output_function = lambda x: fourPeak(x, params, domain=[domain[0], domain[1]])
        else: 
            output_function = lambda x: piecewiseSinusoidal(x, params, domain=[domain[0], domain[1]])

    elif ic == 'monomial': 
        output_function = lambda x: monomial(x, params)
    
    elif ic == 'topHat': 
        output_function = lambda x: topHat(x, params, domain=[domain[0], domain[1]])
    
    elif ic == 'unitPulse':
        output_function = lambda x: unitPulse(x, params, domain=[domain[0], domain[1]])

    elif ic == 'sawtooth': 
        output_function = lambda x: sawtooth(x, params, domain=[domain[0], domain[1]])
    
    elif ic == 'fourPeak': 
        output_function = lambda x: fourPeak(x, params, domain=[domain[0], domain[1]])
    
    elif ic == 'piecewiseSinusoidal':
        output_function = lambda x: piecewiseSinusoidal(x, params, domain=[domain[0], domain[1]])

    elif ic == 'piecewiseConstantSinusoidal':
        output_function = lambda x: piecewiseConstantSinusoidal(x, params, domain=[domain[0], domain[1]])
        
    return output_function 

def monomial(input, params): 
    bias = params['bias']
    coeff = params['coeff']
    power = params['power']
    output = coeff*(-input)**power + 2*bias * np.ones((len(input)))
    return output

def topHat(input, params, disc=0, domain=[-5, 5]):
    bias = params['bias']
    jump = params['jump']
    output = np.ones(len(input))*bias
    for i in range(len(input)):
        if input[i] < domain[0] or input[i] > domain[1]:
            print("Error: Input outside Domain.")
            break
        elif input[i] >=-0.5*domain[1]+disc and input[i] <= 0.5*domain[1]+disc: 
            output[i] += jump
    return output

def unitPulse(input, params, domain=[-5, 5]): 
    bias = params['bias']
    jump = params['jump']
    output = np.ones(len(input))*bias
    for i in range(len(input)):
        if input[i] < -5 or input[i] > 5:
            print("Error: Input outside Domain.")
            break
        elif input[i] >=-0.875*domain[1] and input[i] <= -0.625*domain[1]: 
            output[i] += jump
        elif input[i] >=-0.375*domain[1] and input[i] <= -0.125*domain[1]: 
            output[i] += jump
        elif input[i] >=0.125*domain[1] and input[i] <= 0.375*domain[1]: 
            output[i] += jump
        elif input[i] >=0.625*domain[1] and input[i] <= 0.875*domain[1]: 
            output[i] += jump
        
    return output

def sawtooth(input, params, domain=[-5, 5]): 
    coeff = 0.5
    output = np.ones(len(input))*3.75
    for i in range(len(input)):
        if input[i] < domain[0] or input[i] >domain[1]:
            print("Error: Input outside Domain.")
            break
        elif input[i] >=-1*domain[1]and input[i] <= -0.5*domain[1]: 
            output[i] += 1-coeff*(input[i]-domain[0])
        elif input[i] >=-0.5*domain[1] and input[i] <= 0*domain[1]: 
            output[i] += 1-coeff*(input[i]-0.5*domain[0])
        elif input[i] >=0*domain[1] and input[i] <= 0.5*domain[1]: 
            output[i] += 1-coeff*input[i]
        elif input[i] >=0.5*domain[1] and input[i] <= 1*domain[1]: 
            output[i] += 1-coeff*(input[i]-0.5*domain[1])
    return output

def fourPeak(input, params, domain=[-5, 5]):
    c = 0.5 
    z = -0.7 
    delta = 0.005 
    alpha = 10
    beta = np.log(2)/(36*delta**2)
    bias = params['bias']
    coeff = params['coeff']
    output = np.ones(len(input))*bias
    def F_function(x, alpha, a):
        y = 1-alpha**2*(x-a)**2
        if y > 0: 
            y = np.sqrt(y)
        else:
            y = 0
        return y

    def G_function(x, beta, center):
        y = np.exp(-beta*(x-center)**2)
        return y
    
    for i in range(len(input)): 
        if input[i] >= -0.8*domain[1] and input[i] <= -0.6*domain[1]: 
            output[i] += coeff * ((1/6)*(G_function(input[i], beta, z-delta) + G_function(input[i], beta, z+delta)) + (2/3)* G_function(input[i], beta, z))
        elif input[i] >= -0.4*domain[1] and input[i] <= -0.2*domain[1]:
            output[i] += coeff
        elif input[i] >= 0*domain[1] and input[i] <= 0.2*domain[1]:
            output[i] += coeff * (1 - abs(10*(input[i]-0.1)))
        elif input[i] >= 0.4*domain[1] and input[i] <= 0.6*domain[1]:
            output[i] += coeff * ((1/6)*(F_function(input[i], alpha, c-delta)+F_function(input[i], alpha, c+delta)+4*F_function(input[i], alpha, c)))
        
    return output

def piecewiseSinusoidal(input, params, domain=[-5, 5]): 
    bias = 1
    freq = params['freq']
    amplitude = params['amplitude']
    jump = 2.5
    output = np.ones(len(input))*bias + 0.5*np.sin(freq*np.pi *input/domain[1])
    for i in range(len(input)):
        if input[i] < domain[0] or input[i] > domain[1]:
            print("Error: Input outside Domain.")
            break
        elif input[i] >=-0.6*domain[0] and input[i] <= 0.6*domain[1]: 
            output[i] += jump + amplitude*np.sin(freq*np.pi *input[i])
    return output


def piecewiseConstantSinusoidal(input, params, domain=[-5, 5]): 
    bias = 1
    freq = params['freq']
    amplitude = params['amplitude']
    jump = 2.5
    output = np.ones(len(input))*bias
    for i in range(len(input)):
        if input[i] < domain[0] or input[i] > domain[1]:
            print("Error: Input outside Domain.")
            break
        elif input[i] >=-0.6*domain[1] and input[i] <= 0.6*domain[1]: 
            output[i] += jump + amplitude*np.sin(freq*np.pi *input[i])
    return output

