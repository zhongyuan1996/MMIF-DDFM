import numpy as np
import os
import torch
import torch.fft
# import cv2

def EM_Multimodal_Initial(input_tensor):
    device = input_tensor.device
    kernal1 = torch.tensor([[1,-1]]).to(device)
    kernal2 = torch.tensor([[1],[-1]]).to(device)

    #adding transformation mapping matrix to get gi, li and sv
    trans_gi = torch.ones_like(input_tensor).to(device)
    trans_li = torch.ones_like(input_tensor).to(device)
    trans_sv = torch.ones_like(input_tensor).to(device)

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
        "trans_gi":trans_gi, "trans_li": trans_li, "trans_sv": trans_sv,
        "F1g":F1g, "F2g":F2g, "F1s":F1s, "F2s":F2s,
        "tau_y":tau_y, "tau_g":tau_g, "tau_s":tau_s,
        "fft_g_kernal1":fft_g_kernal1, "fft_g_kernal2":fft_g_kernal2,
        "fft_s_kernal1":fft_s_kernal1, "fft_s_kernal2":fft_s_kernal2,
        "fft_g_kernal1sq":fft_g_kernal1sq, "fft_g_kernal2sq":fft_g_kernal2sq,
        "fft_s_kernal1sq":fft_s_kernal1sq, "fft_s_kernal2sq":fft_s_kernal2sq
    }

    return HP

def EM_Multimodal_onestep(f_pre, A, B, HyperP, lamb=0.5, eta=0.01):
    device = f_pre.device

    fft_g_k1 = HyperP["fft_g_kernal1"].to(device)
    fft_g_k2 = HyperP["fft_g_kernal2"].to(device)
    fft_s_k1 = HyperP["fft_s_kernal1"].to(device)
    fft_s_k2 = HyperP["fft_s_kernal2"].to(device)
    fft_g_k1sq = HyperP["fft_g_kernal1sq"].to(device)
    fft_g_k2sq = HyperP["fft_g_kernal2sq"].to(device)
    fft_s_k1sq = HyperP["fft_s_kernal1sq"].to(device)
    fft_s_k2sq = HyperP["fft_s_kernal2sq"].to(device)

    trans_gi = HyperP["trans_gi"].to(device)
    trans_li = HyperP["trans_li"].to(device)
    trans_sv = HyperP["trans_sv"].to(device)

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

    m_gi = trans_gi * A
    m_li = trans_li * A
    m_sv = trans_sv * B

    Y = m_li - m_gi
    X_G = f_pre - m_gi
    X_S = f_pre - m_sv
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

        trans_gi = ETA*(Y-H_G) / (2*RHO_G**2 * A**2 + ETA * A**2)
        trans_sv = ETA*(Y-H_S) / (2*RHO_S**2 * B**2 + ETA * B**2)
        trans_li = f_pre/A

    F_G = m_gi-X_G
    F_S = m_sv-X_S

    F = (F_G+F_S)/2

    #return F,{"C":C, "D":D, "F2":F2, "F1":F1, "H":H, "tau_a":tau_a, "tau_b":tau_b,
    #"fft_k1":fft_k1, "fft_k2":fft_k2, "fft_k1sq":fft_k1sq, "fft_k2sq":fft_k2sq}
    return F, {"rho_y":rho_y, "rho_g":rho_g, "rho_s":rho_s, "H":H,
        "trans_gi":trans_gi, "trans_li": trans_li, "trans_sv": trans_sv,
        "F1g":F1g, "F2g":F2g, "F1s":F1s, "F2s":F2s,
        "tau_y":tau_y, "tau_g":tau_g, "tau_s":tau_s,
        "fft_g_kernal1":fft_g_k1, "fft_g_kernal2":fft_g_k2,
        "fft_s_kernal1":fft_s_k1, "fft_s_kernal2":fft_s_k2,
        "fft_g_kernal1sq":fft_g_k1sq, "fft_g_kernal2sq":fft_g_k2sq,
        "fft_s_kernal1sq":fft_s_k1sq, "fft_s_kernal2sq":fft_s_k2sq
    }
    
def EM_Initial_gls(IN):
    device = IN.device

    k1 = torch.tensor([[1,-1]]).to(device)
    k2 = torch.tensor([[1],[-1]]).to(device)

    fft_k1_g = psf2otf(k1, np.shape(IN)).to(device)
    fft_k1_l = psf2otf(k1, np.shape(IN)).to(device)
    fft_k1_s = psf2otf(k1, np.shape(IN)).to(device)
    fft_k2_g = psf2otf(k2, np.shape(IN)).to(device)
    fft_k2_l = psf2otf(k2, np.shape(IN)).to(device)
    fft_k2_s = psf2otf(k2, np.shape(IN)).to(device)
    fft_k1sq_g = torch.conj(fft_k1_g)*fft_k1_g.to(device)
    fft_k1sq_l = torch.conj(fft_k1_l)*fft_k1_l.to(device)
    fft_k1sq_s = torch.conj(fft_k1_s)*fft_k1_s.to(device)
    fft_k2sq_g = torch.conj(fft_k2_g)*fft_k2_g.to(device)
    fft_k2sq_l = torch.conj(fft_k2_l)*fft_k2_l.to(device)
    fft_k2sq_s = torch.conj(fft_k2_s)*fft_k2_s.to(device)

    sig_g = torch.ones_like(IN).to(device)
    sig_l = torch.ones_like(IN).to(device)
    sig_s = torch.ones_like(IN).to(device)
    F1g = torch.zeros_like(IN).to(device)
    F2g = torch.zeros_like(IN).to(device)
    F1l = torch.zeros_like(IN).to(device)
    F2l = torch.zeros_like(IN).to(device)
    F1s = torch.zeros_like(IN).to(device)
    F2s = torch.zeros_like(IN).to(device)
    H_G = torch.zeros_like(IN).to(device)
    H_L = torch.zeros_like(IN).to(device)
    H_S = torch.zeros_like(IN).to(device)

    g=torch.ones_like(IN).to(device)
    l=torch.ones_like(IN).to(device)
    s=torch.ones_like(IN).to(device)

    # g=torch.randn_like(IN).to(device)
    # l=torch.randn_like(IN).to(device)
    # s=torch.randn_like(IN).to(device)
    #not because of this

    tau_g = 1
    tau_l = 1
    tau_s = 1

    HP={"g":g, "l":l, "s":s,
        "sig_g":sig_g, "sig_l":sig_l, "sig_s":sig_s,
        "F1g":F1g, "F2g":F2g, "F1l":F1l, "F2l":F2l, "F1s":F1s, "F2s":F2s,
        "H_G":H_G, "H_L":H_L, "H_S":H_S,
        "tau_g":tau_g, "tau_l":tau_l, "tau_s":tau_s,
        "fft_k1_g":fft_k1_g, "fft_k1_l":fft_k1_l, "fft_k1_s":fft_k1_s,
        "fft_k2_g":fft_k2_g, "fft_k2_l":fft_k2_l, "fft_k2_s":fft_k2_s,
        "fft_k1sq_g":fft_k1sq_g, "fft_k1sq_l":fft_k1sq_l, "fft_k1sq_s":fft_k1sq_s,
        "fft_k2sq_g":fft_k2sq_g, "fft_k2sq_l":fft_k2sq_l, "fft_k2sq_s":fft_k2sq_s
    }

    return HP

def EM_onestep_gls(f_pre, I, V, HyperP, lamb=0.5, rho=0.01):
    device = f_pre.device

    fft_k1_g = HyperP["fft_k1_g"].to(device)
    fft_k1_l = HyperP["fft_k1_l"].to(device)
    fft_k1_s = HyperP["fft_k1_s"].to(device)
    fft_k2_g = HyperP["fft_k2_g"].to(device)
    fft_k2_l = HyperP["fft_k2_l"].to(device)
    fft_k2_s = HyperP["fft_k2_s"].to(device)
    fft_k1sq_g = HyperP["fft_k1sq_g"].to(device)
    fft_k1sq_l = HyperP["fft_k1sq_l"].to(device)
    fft_k1sq_s = HyperP["fft_k1sq_s"].to(device)
    fft_k2sq_g = HyperP["fft_k2sq_g"].to(device)
    fft_k2sq_l = HyperP["fft_k2sq_l"].to(device)
    fft_k2sq_s = HyperP["fft_k2sq_s"].to(device)

    sig_g = HyperP["sig_g"].to(device)
    sig_l = HyperP["sig_l"].to(device)
    sig_s = HyperP["sig_s"].to(device)
    F1g = HyperP["F1g"].to(device)
    F2g = HyperP["F2g"].to(device)
    F1l = HyperP["F1l"].to(device)
    F2l = HyperP["F2l"].to(device)
    F1s = HyperP["F1s"].to(device)
    F2s = HyperP["F2s"].to(device)
    H_G = HyperP["H_G"].to(device)
    H_L = HyperP["H_L"].to(device)
    H_S = HyperP["H_S"].to(device)
    g = HyperP["g"].to(device)
    l = HyperP["l"].to(device)
    s = HyperP["s"].to(device)
    tau_g = HyperP["tau_g"]
    tau_l = HyperP["tau_l"]
    tau_s = HyperP["tau_s"]

    LAMBDA = lamb
    #e-step
    # sig_g = torch.sqrt(2/tau_g/((g*I)**2+1e-6))
    # sig_l = torch.sqrt(2/tau_l/((l*I)**2+1e-6))
    # sig_s = torch.sqrt(2/tau_s/((s*V)**2+1e-6))
    # not because of this

    sig_g = torch.sqrt(2/tau_g/((f_pre-g*I)**2+1e-6))
    sig_l = torch.sqrt(2/tau_l/((f_pre-l*I)**2+1e-6))
    sig_s = torch.sqrt(2/tau_s/((f_pre-s*V)**2+1e-6))

    # sig_g[sig_g>2*sig_l] = 2*sig_l[sig_g>2*sig_l]
    # sig_s[sig_s>2*sig_l] = 2*sig_l[sig_s>2*sig_l]
    #not because of this

    RHO = rho

    tau_g = 1./(sig_g+1e-10)+tau_g/2
    tau_g = torch.mean(tau_g)
    tau_l = 1./(sig_l+1e-10)+tau_l/2
    tau_l = torch.mean(tau_l)
    tau_s = 1./(sig_s+1e-10)+tau_s/2
    tau_s = torch.mean(tau_s)

    for s in range(1):
        H_G = prox_tv(f_pre-g*I,F1g,F2g, fft_k1_g, fft_k2_g, fft_k1sq_g, fft_k2sq_g)
        H_L = prox_tv(f_pre-l*I,F1l,F2l, fft_k1_l, fft_k2_l, fft_k1sq_l, fft_k2sq_l)
        H_S = prox_tv(f_pre-s*V,F1s,F2s, fft_k1_s, fft_k2_s, fft_k1sq_s, fft_k2sq_s)
        #this has no change, still a unconditional one
        # H_G = prox_tv(g*I,F1g,F2g, fft_k1_g, fft_k2_g, fft_k1sq_g, fft_k2sq_g)
        # H_L = prox_tv(l*I,F1l,F2l, fft_k1_l, fft_k2_l, fft_k1sq_l, fft_k2sq_l)
        # H_S = prox_tv(s*V,F1s,F2s, fft_k1_s, fft_k2_s, fft_k1sq_s, fft_k2sq_s)
        #this generate a unconditional one

        a1g=torch.zeros_like(H_G)
        a1g[:,:,:,:-1]=H_G[:,:,:,:-1]-H_G[:,:,:,1:]
        a1g[:,:,:,-1]=H_G[:,:,:,-1]
        F1g = (RHO/(2*LAMBDA+RHO))*a1g

        a2g=torch.zeros_like(H_G)
        a2g[:,:,:-1,:]=H_G[:,:,:-1,:]-H_G[:,:,1:,:]
        a2g[:,:,-1,:]=H_G[:,:,-1,:]
        F2g = (RHO/(2*LAMBDA+RHO))*a2g

        a1l=torch.zeros_like(H_L)
        a1l[:,:,:,:-1]=H_L[:,:,:,:-1]-H_L[:,:,:,1:]
        a1l[:,:,:,-1]=H_L[:,:,:,-1]
        F1l = (RHO/(2*LAMBDA+RHO))*a1l

        a2l=torch.zeros_like(H_L)
        a2l[:,:,:-1,:]=H_L[:,:,:-1,:]-H_L[:,:,1:,:]
        a2l[:,:,-1,:]=H_L[:,:,-1,:]
        F2l = (RHO/(2*LAMBDA+RHO))*a2l

        a1s=torch.zeros_like(H_S)
        a1s[:,:,:,:-1]=H_S[:,:,:,:-1]-H_S[:,:,:,1:]
        a1s[:,:,:,-1]=H_S[:,:,:,-1]
        F1s = (RHO/(2*LAMBDA+RHO))*a1s

        a2s=torch.zeros_like(H_S)
        a2s[:,:,:-1,:]=H_S[:,:,:-1,:]-H_S[:,:,1:,:]
        a2s[:,:,-1,:]=H_S[:,:,-1,:]
        F2s = (RHO/(2*LAMBDA+RHO))*a2s

        # g = (2*sig_g**2*f_pre*I+RHO*(H_G)*I - RHO*I*f_pre)/(2*sig_g**2*I**2+RHO*I**2)
        # l = (2*sig_l**2*f_pre*I+RHO*(H_L)*I - RHO*I*f_pre)/(2*sig_l**2*I**2+RHO*I**2)
        # s = (2*sig_s**2*f_pre*V+RHO*(H_S)*V - RHO*V*f_pre)/(2*sig_s**2*V**2+RHO*V**2)
        #not because of this

        g = (2*sig_g**2*f_pre*I+RHO*(f_pre-H_G)*I - RHO*I*f_pre)/(2*sig_g**2*I**2+RHO*I**2)
        l = (2*sig_l**2*f_pre*I+RHO*(f_pre-H_L)*I - RHO*I*f_pre)/(2*sig_l**2*I**2+RHO*I**2)
        s = (2*sig_s**2*f_pre*V+RHO*(f_pre-H_S)*V - RHO*V*f_pre)/(2*sig_s**2*V**2+RHO*V**2)
        #also not because of this

    F= I + V + f_pre - g*I - l*I - s*V
    # F = I+V
    # The model basically outputs f_pre, and g*I, l*I and s*V are all basically equals to f_pre

    # print('s after estep')
    # print(s)

    return F, {"g":g, "l":l, "s":s,
        "sig_g":sig_g, "sig_l":sig_l, "sig_s":sig_s,
        "F1g":F1g, "F2g":F2g, "F1l":F1l, "F2l":F2l, "F1s":F1s, "F2s":F2s,
        "H_G":H_G, "H_L":H_L, "H_S":H_S,
        "tau_g":tau_g, "tau_l":tau_l, "tau_s":tau_s,
        "fft_k1_g":fft_k1_g, "fft_k1_l":fft_k1_l, "fft_k1_s":fft_k1_s,
        "fft_k2_g":fft_k2_g, "fft_k2_l":fft_k2_l, "fft_k2_s":fft_k2_s,
        "fft_k1sq_g":fft_k1sq_g, "fft_k1sq_l":fft_k1sq_l, "fft_k1sq_s":fft_k1sq_s,
        "fft_k2sq_g":fft_k2sq_g, "fft_k2sq_l":fft_k2sq_l, "fft_k2sq_s":fft_k2sq_s
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
        # H = prox_tv(Y,F1,F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)
        #Y-X, X or Y does not matter at all
        H = prox_tv(Y-X,F1,F2, fft_k1, fft_k2, fft_k1sq, fft_k2sq)


        a1=torch.zeros_like(H)
        a1[:,:,:,:-1]=H[:,:,:,:-1]-H[:,:,:,1:]
        a1[:,:,:,-1]=H[:,:,:,-1]
        #F1, F2 are delta k
        F1 = (RHO/(2*LAMBDA+RHO))*a1

        a2=torch.zeros_like(H)
        a2[:,:,:-1,:]=H[:,:,:-1,:]-H[:,:,1:,:]
        a2[:,:,-1,:]=H[:,:,-1,:]
        F2 = (RHO/(2*LAMBDA+RHO))*a2
        #Y-H is k
        #X = (2*C*Y+RHO*(H))/(2*C+2*D+RHO)
        #does not matter Y-H or H
        X = (2*C*Y+RHO*(Y-H))/(2*C+2*D+RHO)


    F = I-X
    # F = X + V

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
    
