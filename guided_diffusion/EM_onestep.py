import numpy as np
import os
import torch
import torch.fft
# import cv2

def EM_Multimodal_Initial(input_tensor):
    device = input_tensor.device
    kernal1 = torch.tensor([[1,-1]]).to(device)
    kernal2 = torch.tensor([[1],[-1]]).to(device)

    fft_g_kernal1 = psf2otf(kernal1, np.shape(input_tensor)).to(device)
    fft_g_kernal2 = psf2otf(kernal2, np.shape(input_tensor)).to(device)
    fft_s_kernal1 = psf2otf(kernal1, np.shape(input_tensor)).to(device)
    fft_s_kernal2 = psf2otf(kernal2, np.shape(input_tensor)).to(device)
    fft_g_kernal1sq = torch.conj(fft_g_kernal1)*fft_g_kernal1.to(device)
    fft_g_kernal2sq = torch.conj(fft_g_kernal2)*fft_g_kernal2.to(device)
    fft_s_kernal1sq = torch.conj(fft_s_kernal1)*fft_s_kernal1.to(device)
    fft_s_kernal2sq = torch.conj(fft_s_kernal2)*fft_s_kernal2.to(device)

    rho_y = torch.ones_like(input_tensor).to(device)
    rho_g = torch.ones_like(input_tensor).to(device)
    rho_s = torch.ones_like(input_tensor).to(device)
    F1g = torch.zeros_like(input_tensor).to(device)
    F2g = torch.zeros_like(input_tensor).to(device)
    F1s = torch.zeros_like(input_tensor).to(device)
    F2s = torch.zeros_like(input_tensor).to(device)
    H = torch.zeros_like(input_tensor).to(device)
    tau_y = 1
    tau_g = 1
    tau_s = 1
    HP={
        "rho_y":rho_y, "rho_g":rho_g, "rho_s":rho_s, "H":H,
        "F1g":F1g, "F2g":F2g, "F1s":F1s, "F2s":F2s,
        "tau_y":tau_y, "tau_g":tau_g, "tau_s":tau_s,
        "fft_g_kernal1":fft_g_kernal1, "fft_g_kernal2":fft_g_kernal2,
        "fft_s_kernal1":fft_s_kernal1, "fft_s_kernal2":fft_s_kernal2,
        "fft_g_kernal1sq":fft_g_kernal1sq, "fft_g_kernal2sq":fft_g_kernal2sq,
        "fft_s_kernal1sq":fft_s_kernal1sq, "fft_s_kernal2sq":fft_s_kernal2sq
    }

    return HP

def EM_Multimodal_onestep(f_pre, m_g, m_l, m_s, HyperP, lamb=0.5, eta=0.01):
    device = f_pre.device

    fft_g_k1 = HyperP["fft_g_kernal1"].to(device)
    fft_g_k2 = HyperP["fft_g_kernal2"].to(device)
    fft_s_k1 = HyperP["fft_s_kernal1"].to(device)
    fft_s_k2 = HyperP["fft_s_kernal2"].to(device)
    fft_g_k1sq = HyperP["fft_g_kernal1sq"].to(device)
    fft_g_k2sq = HyperP["fft_g_kernal2sq"].to(device)
    fft_s_k1sq = HyperP["fft_s_kernal1sq"].to(device)
    fft_s_k2sq = HyperP["fft_s_kernal2sq"].to(device)

    rho_y = HyperP["rho_y"].to(device)
    rho_g = HyperP["rho_g"].to(device)
    rho_s = HyperP["rho_s"].to(device)
    F1g = HyperP["F1g"].to(device)
    F2g = HyperP["F2g"].to(device)
    F1s = HyperP["F1s"].to(device)
    F2s = HyperP["F2s"].to(device)

    H = HyperP["H"].to(device)
    tau_y = HyperP["tau_y"]
    tau_g = HyperP["tau_g"]
    tau_s = HyperP["tau_s"]

    LAMBDA = lamb
    Y = m_l - m_g
    X_G = f_pre - m_g
    X_S = f_pre - m_s
    #e-step
    RHO_G = torch.sqrt(2/tau_g/(X_G**2+1e-6))
    RHO_Y = torch.sqrt(2/tau_y/((Y-X_G)**2+1e-6))
    RHO_S = torch.sqrt(2/tau_s/(X_S**2+1e-6))
    RHO_G[RHO_G>2*RHO_Y] = 2*RHO_Y[RHO_G>2*RHO_Y]
    RHO_S[RHO_S>2*RHO_Y] = 2*RHO_Y[RHO_S>2*RHO_Y]
    ETA=eta

    tau_g = 1./(RHO_G+1e-10)+tau_g/2
    tau_g = torch.mean(tau_g)
    tau_y = 1./(RHO_Y+1e-10)+tau_y/2
    tau_y = torch.mean(tau_y)
    tau_s = 1./(RHO_S+1e-10)+tau_s/2
    tau_s = torch.mean(tau_s)

    # m-step
    for s in range(1):
        H_G = prox_tv(Y-X_G,F1g,F2g, fft_g_k1, fft_g_k2, fft_g_k1sq, fft_g_k2sq)
        H_S = prox_tv(Y-X_S,F1s,F2s, fft_s_k1, fft_s_k2, fft_s_k1sq, fft_s_k2sq)

        #H = prox_tv(Y-X,F1,F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)
        a1g=torch.zeros_like(H_G)
        a1g[:,:,:,:-1]=H_G[:,:,:,:-1]-H_G[:,:,:,1:]
        a1g[:,:,:,-1]=H_G[:,:,:,-1]
        F1g = (RHO_G/(2*LAMBDA+ETA))*a1g

        a2g=torch.zeros_like(H_G)
        a2g[:,:,:-1,:]=H_G[:,:,:-1,:]-H_G[:,:,1:,:]
        a2g[:,:,-1,:]=H_G[:,:,-1,:]
        F2g = (RHO_G/(2*LAMBDA+ETA))*a2g
        X_G = (2*RHO_Y*Y+ETA*(Y-H_G))/(2*RHO_Y+2*RHO_G+ETA)

        a1s=torch.zeros_like(H_S)
        a1s[:,:,:,:-1]=H_S[:,:,:,:-1]-H_S[:,:,:,1:]
        a1s[:,:,:,-1]=H_S[:,:,:,-1]
        F1s = (RHO_S/(2*LAMBDA+ETA))*a1s

        a2s=torch.zeros_like(H_S)
        a2s[:,:,:-1,:]=H_S[:,:,:-1,:]-H_S[:,:,1:,:]
        a2s[:,:,-1,:]=H_S[:,:,-1,:]
        F2s = (RHO_S/(2*LAMBDA+ETA))*a2s
        X_S = (ETA*(Y-H_S))/(2*RHO_S+ETA)

    F_G = m_g-X_G
    F_S = m_s-X_S

    F = (F_G+F_S)/2

    #return F,{"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    #"fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    return F, {"rho_y":rho_y, "rho_g":rho_g, "rho_s":rho_s, "H":H,
        "F1g":F1g, "F2g":F2g, "F1s":F1s, "F2s":F2s,
        "tau_y":tau_y, "tau_g":tau_g, "tau_s":tau_s,
        "fft_g_kernal1":fft_g_k1, "fft_g_kernal2":fft_g_k2,
        "fft_s_kernal1":fft_s_k1, "fft_s_kernal2":fft_s_k2,
        "fft_g_kernal1sq":fft_g_k1sq, "fft_g_kernal2sq":fft_g_k2sq,
        "fft_s_kernal1sq":fft_s_k1sq, "fft_s_kernal2sq":fft_s_k2sq
    }
    
def EM_Initial(IR):
    device = IR.device

    k1 = torch.tensor([[1,-1]]).to(device)
    k2 = torch.tensor([[1],[-1]]).to(device)

    fft_k1 = psf2otf(k1, np.shape(IR)).to(device)
    fft_k2 = psf2otf(k2, np.shape(IR)).to(device)
    fft_k1sq = torch.conj(fft_k1)*fft_k1.to(device)
    fft_k2sq = torch.conj(fft_k2)*fft_k2.to(device)

    C  = torch.ones_like(IR).to(device)
    D = torch.ones_like(IR).to(device)
    F2 = torch.zeros_like(IR).to(device)
    F1 = torch.zeros_like(IR).to(device)
    H  = torch.zeros_like(IR).to(device)
    tau_a = 1
    tau_b = 1
    HP={"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    return HP

def EM_onestep(f_pre,I,V,HyperP,lamb=0.5,rho=0.01): 
    device = f_pre.device

    fft_k1 = HyperP["fft_k1"].to(device)
    fft_k2 = HyperP["fft_k2"].to(device)
    fft_k1sq = HyperP["fft_k1sq"].to(device)
    fft_k2sq = HyperP["fft_k2sq"].to(device)
    C  = HyperP["C"].to(device)
    D  = HyperP["D"].to(device)
    F2 = HyperP["F2"].to(device)
    F1 = HyperP["F1"].to(device)
    H  = HyperP["H"].to(device)
    tau_a = HyperP["tau_a"]#.to(device)
    tau_b = HyperP["tau_b"]#.to(device)

    LAMBDA =  lamb

    Y = I - V
    X = f_pre - V
    #e-step
    D = torch.sqrt(2/tau_b/(X**2+1e-6))
    C = torch.sqrt(2/tau_a/((Y-X+1e-6)**2))
    D[D>2*C] = 2*C[D>2*C]
    RHO =rho # .5*(C+D)
    
    tau_b = 1./(D+1e-10)+tau_b/2
    tau_b = torch.mean(tau_b)
    tau_a = 1./(C+1e-10)+tau_a/2
    tau_a = torch.mean(tau_a)

    # m-step
    for s in range(1):
        H = prox_tv(Y-X,F1,F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)

        a1=torch.zeros_like(H)
        a1[:,:,:,:-1]=H[:,:,:,:-1]-H[:,:,:,1:]
        a1[:,:,:,-1]=H[:,:,:,-1]
        F1 = (RHO/(2*LAMBDA+RHO))*a1

        a2=torch.zeros_like(H)
        a2[:,:,:-1,:]=H[:,:,:-1,:]-H[:,:,1:,:]
        a2[:,:,-1,:]=H[:,:,-1,:]
        F2 = (RHO/(2*LAMBDA+RHO))*a2
        X = (2*C*Y+RHO*(Y-H))/(2*C+2*D+RHO)


    F = I-X

    return F,{"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    "fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}

def prox_tv(X, F1, F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq):
    fft_X = torch.fft.fft2(X)
    fft_F1 = torch.fft.fft2(F1)
    fft_F2 = torch.fft.fft2(F2)

    H = fft_X + torch.conj(fft_k1)*fft_F1 + torch.conj(fft_k2)*fft_F2
    H = H/( 1+fft_k1sq+fft_k2sq)
    H = torch.real(torch.fft.ifft2(H))
    return H

def psf2otf(psf, outSize):
    psfSize = torch.tensor(psf.shape)
    outSize = torch.tensor(outSize[-2:])
    padSize = outSize - psfSize
    psf=torch.nn.functional.pad(psf,(0, padSize[1],0, padSize[0]), 'constant')

    for i in range(len(psfSize)):
        psf = torch.roll(psf, -int(psfSize[i] / 2), i)
    otf = torch.fft.fftn(psf)
    nElem = torch.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * torch.log2(psfSize[k]) * nffts
    if torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= nOps * torch.finfo(torch.float32).eps:
        otf = torch.real(otf)
    return otf
    
