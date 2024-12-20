import numpy as np
def Q(point,centre,ro):
    x = point[0,0] - centre[0,0]
    y = point[0,1] - centre[0,1]
    Qo = 5 ## amplitude in W/mm^2 
    return Qo*np.exp(-(x**2+y**2)/ro**2)  ## W/mm^3

def k_T(T):
    return 3.276e-7*T+0.0793  #W/mm.K

def cp_T(T):
    return 0*T+550 #J/kgK

def rho_T(T):
    return 0*T+2.116e-6 #kg/mm^3

def rho_Ti(T, phase = 'alpha'):
    if phase == 'alpha':
        return -5.13e-5*(T**2)-0.01935*T+4451
    elif phase == 'beta':
        return -2.762e-6*(T**2)-0.1663*T+4468
    elif phase == 'liquid':
        return -0.565*T+5093
    else:
        return T

def cp_Ti(T,phase = 'alpha'):
    if phase == 'alpha':
        return 0.25*T+483
    elif phase == 'beta':
        return 0.14*T+530
    elif phase == 'liquid':
        return 930
    else:
        return T

def k_Ti(T,phase = 'alpha'):
    if phase == 'alpha':
        return 0.012*T+3.3
    elif phase == 'beta':
        return 0.016*T-3
    elif phase == 'liquid':
        return 0.0175*T-4.5
    else:
        return T

def props_chooser(T, T_rep, process = 'heating'):
    if (T_rep<1268 and process == 'heating') or (T_rep<=1073 and process == 'cooling'):
        return rho_Ti(T,phase = 'alpha'),cp_Ti(T,phase = 'alpha'),k_Ti(T,phase = 'alpha')
    elif (T_rep<1928 and process == 'heating') or (T_rep>1073 and process == 'cooling'):
        return rho_Ti(T,phase = 'beta'),cp_Ti(T,phase = 'beta'),k_Ti(T,phase = 'beta')
    elif (T_rep>=1928 and process == 'heating') or (T_rep>=1878 and process == 'cooling'):
        return rho_Ti(T,phase = 'liquid'),cp_Ti(T,phase = 'liquid'),k_Ti(T,phase = 'liquid')
    else:
        return -1
   
def matrix_helper(args):
    nodes,ele,centre,theta_prev_time,theta_prev_pic,mode = args
    gp = 3
    
    K_row,K_col,K_data = [],[],[]
    G_row,G_col,G_data = [],[],[]
    F_row,F_data = [],[]
    BT_row,BT_data = [],[]
    
    qo = 1e-3   # W/mm^2
    c = 658 #J/kg.K
    rho = 7.6e-6 #kg/mm^3
    kappa = 0.025 #W/mm.K
    ro = 2 #mm
    vo = 2 #mm/s
    
    ln = np.where(nodes[:,0] == 0)[0]
    rn = np.where(nodes[:,0] == np.max(nodes[:,0]))[0]
    bn = np.where(nodes[:,1] == 0)[0]
    tn = np.where(nodes[:,1] == np.max(nodes[:,1]))[0]

    data_line = {"ips":{2:[-1/np.sqrt(3),1/np.sqrt(3)],3:[-np.sqrt(3/5),0,np.sqrt(3/5)]},\
    "weights":{2:[1,1],3:[5/9,8/9,5/9]}}
    data_tle = {"ips":{1:[[1/3,1/3]],3:[[1/6,1/6],[1/6,2/3],[2/3,1/6]]},\
       "weights":{1:[1/2],3:[1/6,1/6,1/6]}}

    ips = np.array(data_tle["ips"][gp])
    weights = np.array(data_tle["weights"][gp])
    econ = ele-1
    nnode = econ.shape[0]
    boundary = nodes[np.ix_(econ,[0,1])]
    dN = np.array([[-1,1,0],[-1,0,1]])
    Jac = np.matmul(dN,boundary)
    if np.linalg.det(Jac)<0:
        econ[0],econ[1] = econ[1],econ[0] #reordering for the direction to be counter clockwise
        boundary = nodes[np.ix_(econ,[0,1])] 
        Jac = np.matmul(dN,boundary)
        
    dN_phy = np.linalg.inv(Jac)@dN#2x3
    dN_dx = dN_phy[0,:].reshape(1,-1)

    Jac_inv = np.linalg.inv(Jac)
    
    area = 0
    if mode == "phase_change":
        T_rep = np.mean(theta_prev_time[np.ix_(econ,[0])]) #temperature at the centroid of the element
    K_loc = np.zeros((nnode,nnode))
    G_loc = np.zeros((nnode,nnode))
    f_loc = np.zeros((nnode,nnode))

    for k,ipk in enumerate(ips):
        N = np.array([[(1-ipk[0]-ipk[1]), ipk[0],ipk[1]]])
        a = (Jac_inv@dN).T@(Jac_inv@dN)*(np.linalg.det(Jac))*weights[k]
        b = N.T@dN_dx*(np.linalg.det(Jac))*weights[k]

        if mode == "non_linear":
            # rhos,cs,ks = props_chooser(theta_prev_pic[np.ix_(econ,[0])],T_rep)
            kappa = N@k_T(theta_prev_pic[np.ix_(econ,[0])])
            rho = N@rho_T(theta_prev_pic[np.ix_(econ,[0])])  
            c = N@cp_T(theta_prev_pic[np.ix_(econ,[0])])
        elif mode == "phase_change":
            rhos,cs,kappas = props_chooser(theta_prev_pic[np.ix_(econ,[0])],T_rep)
            kappa = N@kappas/1e3
            rho = N@rhos/1e9 
            c = N@cs
        K_loc += kappa*a
        G_loc += rho*c*vo*b
        X  =np.matmul(N,boundary)
        f_loc += N*Q(X,centre,ro)*np.linalg.det(Jac)*weights[k]
        area += np.linalg.det(Jac)*weights[k]    
    for i in range(nnode):
        if (f_loc.T[i,0]):
            F_row.append(econ[i])
            F_data.append(f_loc.T[i,0])
        for j in range(nnode):
            if K_loc[i][j]!=0:
                K_row.append(econ[i])
                K_col.append(econ[j])
                K_data.append(K_loc[i][j])
                
            if G_loc[i][j]!=0:
                G_row.append(econ[i])
                G_col.append(econ[j])
                G_data.append(G_loc[i][j])

    for l,m in zip([0,1,2],[1,2,0]):
        n1 = econ[l]
        n2 = econ[m]
        check_rn = (n1 in rn and n2 in rn)
        check_tn = (n1 in tn and n2 in tn)
        check_bn = (n1 in bn and n2 in bn)

        bt = np.zeros((2,1))
        if check_rn or check_tn or check_bn:
            line_gp = 3
            line_ips = np.array(data_line["ips"][line_gp])
            line_weights = np.array(data_line["weights"][line_gp])
            for k,ipk in enumerate(line_ips):
                N_line = np.array([(1-ipk)/2, (1+ipk)/2]).reshape(1,-1)
                dN_line = np.array([-1/2, 1/2]).reshape(1,-1)
                line_boundary = nodes[np.ix_([n1,n2],[1 if check_rn else 0])] #assuming interfacial lines are along x axis
                Jac_line = np.matmul(dN_line,line_boundary)
                if np.linalg.det(Jac_line)<0:
                    n1,n2 = n2,n1
                    line_boundary = nodes[np.ix_([n1,n2],[1 if check_rn else 0])] #interchanging nodes"
                    Jac_line = np.matmul(dN_line,line_boundary)
                bt += N_line.T*np.linalg.det(Jac_line)*(-qo)*line_weights[k]
            BT_row.append(n1)
            BT_row.append(n2)
            BT_data.append(bt[0,0])
            BT_data.append(bt[1,0])
                
    return K_row, K_col, K_data, G_row, G_col, G_data, F_row,F_data, BT_row, BT_data,area