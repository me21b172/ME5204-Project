import numpy as np
def Q(point):
    x = point[0,0] #in mm
    return 15e-3*(x/100)*(1-x/100) ## W/mm^3

def k_T(T):
    return (100+0.004*(T-50-273)**2)*1e-3  ##W/mmK
   
def picard_helper(args):
    nodes,ele,theta_prev_time,theta_prev_pic,mode,scheme = args
    gp = 3
    
    K_row,K_col,K_data = [],[],[]
    M_row,M_col,M_data = [],[],[]
    F_row,F_data = [],[]
    
    qo = 0   # W/mm^2
    c = 465 #J/kg.K
    rho = 7e-6 #kg/mm^3
    
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
        
    Jac_inv = np.linalg.inv(Jac)
    
    area = 0
    if mode == "non_linear":
        T_rep = np.mean(theta_prev_time[np.ix_(econ,[0])]) #temperature at the centroid of the element
    K_loc = np.zeros((nnode,nnode))
    M_loc = np.zeros((nnode,nnode))
    f_loc = np.zeros((nnode,nnode))

    for k,ipk in enumerate(ips):
        N = np.array([[(1-ipk[0]-ipk[1]), ipk[0],ipk[1]]])
        a = (Jac_inv@dN).T@(Jac_inv@dN)*(np.linalg.det(Jac))*weights[k]
        b = (N.T@N)*(np.linalg.det(Jac))*weights[k]

        if mode == "non_linear":
            kappa = N@k_T(theta_prev_pic[np.ix_(econ,[0])])

        K_loc += kappa*a
        M_loc += rho*c*b
        X  =np.matmul(N,boundary)
        f_loc += N*Q(X)*np.linalg.det(Jac)*weights[k]
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
            if M_loc[i][j]!=0:
                M_row.append(econ[i])
                M_col.append(econ[j])
                M_data.append(M_loc[i][j])
                

                
    return K_row, K_col, K_data, M_row, M_col, M_data, F_row,F_data,area
