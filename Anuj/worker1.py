import numpy as np
def Q(point,centre,ro):
    x = point[0,0] - centre[0,0]
    y = point[0,1] - centre[0,1]
    Qo = 5 ## amplitude in W/mm^2 
    return Qo*np.exp(-(x**2+y**2)/ro**2)  ## W/mm^3
    # x = point[0,0]
    # return 15e-3*(x/100)*(1-x/100)

def k_T(T):
    # return 3.276e-7*T+0.0793  #W/mm.K
    return (100+0.004*(T-50-273)**2)*1e-3

def cp_T(T):
    # return 550 #J/kgK
    return 465

def rho_T(T):
    return 7e-6 #kg/mm^3

def rho_Ti(T,phase = 'alpha'):
    if phase == 'alpha':
        return -5.13e-5*(T**2)-0.01935*T+4451
    elif phase == 'beta':
        return -2.762e-6*(T**2)-0.1663*T+4468
    elif phase == 'liquid':
        return -0.565*T+5093
    else:
        return T

def cp_Ti(T,process = 'heating',phase = 'alpha'):
    if phase == 'alpha':
        lin = 0.25*T+483
        heat = 13000*np.exp(-0.5*(((T-1160)/90)**2))/(90*np.sqrt(2*np.pi))
        cool = 13000*np.exp(-0.5*(((T-952)/90)**2))/(90*np.sqrt(2*np.pi))
        if process == 'heating':
            return lin+heat
        elif process == 'cooling':
            return lin+cool
    elif phase == 'beta':
        lin = 0.14*T+530
        heat = 41650*np.exp(-0.5*(((T-1905)/9)**2))/(9*np.sqrt(2*np.pi))
        cool = 41650*np.exp(-0.5*(((T-1855)/9)**2))/(9*np.sqrt(2*np.pi))
        if process == 'heating':
            return lin+heat
        elif process == 'cooling':
            return lin+cool
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


def props_chooser(T, phase = 'alpha',process = 'heating'):
    return rho_Ti(T),cp_Ti(T,process,phase),k_Ti(T,phase)

phase_map = {0:'alpha',1:'beta',2:'liquid'}
def phase_determiner(T_rep,process = 'heating'):
    if (T_rep<1268 and process == 'heating') or (T_rep<=1073 and process == 'cooling'):
        return 'alpha'
    elif (T_rep<1928 and process == 'heating') or (T_rep>1073 and process == 'cooling'):
        return 'beta'
    elif (T_rep>=1928 and process == 'heating') or (T_rep>=1878 and process == 'cooling'):
        return 'liquid'
    else:
        return -1
   
def matrix_helper(args):
    nodes,ele,centre,theta_pprev_time,theta_prev_time,theta_prev_pic,mode = args
    gp = 3
    del_t = 1e-50

    K_row,K_col,K_data = [],[],[]
    K_D_row,K_D_col,K_D_data = [],[],[]
    G_row,G_col,G_data = [],[],[]
    M_row,M_col,M_data = [],[],[]
    M_D_row,M_D_col,M_D_data = [],[],[]
    F_row,F_data = [],[]
    BT_row,BT_data = [],[]
    
    # qo = 0   # W/mm^2
    qo = 1e-3   # W/mm^2
    # c = 658 #J/kg.K
    # rho = 7.6e-6 #kg/mm^3
    # kappa = 0.025 #W/mm.K
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
    # if mode == "phase_change":
    T_rep = np.mean(theta_prev_time[np.ix_(econ,[0])]) #temperature at the centroid of the element
    K_loc = np.zeros((nnode,nnode))
    K_D_loc = np.zeros((nnode,nnode))
    G_loc = np.zeros((nnode,nnode))
    M_loc = np.zeros((nnode,nnode))
    M_D_loc = np.zeros((nnode,nnode))
    f_loc = np.zeros((nnode,nnode))

    for k,ipk in enumerate(ips):
        N = np.array([[(1-ipk[0]-ipk[1]), ipk[0],ipk[1]]])
        a = (Jac_inv@dN).T@(Jac_inv@dN)*(np.linalg.det(Jac))*weights[k]
        b = N.T@dN_dx*(np.linalg.det(Jac))*weights[k]
        m = N.T@N*(np.linalg.det(Jac))*weights[k]

        if mode == "non_linear":
            # rhos,cs,ks = props_chooser(theta_prev_pic[np.ix_(econ,[0])],T_rep)
            kappa = k_T((N@theta_prev_pic[np.ix_(econ,[0])])[0][0])
            rho = rho_T((N@theta_prev_pic[np.ix_(econ,[0])])[0][0])  
            c = cp_T((N@theta_prev_pic[np.ix_(econ,[0])])[0][0])
            kappa_t = k_T((N@(theta_prev_pic[np.ix_(econ,[0])]))[0][0]+del_t)
            rho_t = rho_T((N@theta_prev_pic[np.ix_(econ,[0])])[0][0]+del_t)
            c_t = cp_T((N@theta_prev_pic[np.ix_(econ,[0])])[0][0]+del_t)
            # del_k_del_t = (kappa_t - kappa)/del_t
            # del_rho_del_t = (rho_t - rho)/del_t
            # del_c_del_t = (c_t - c)/del_t   
        elif mode == "phase_change":
            process = 'heating'
            T_rep_prev = np.mean(theta_pprev_time[np.ix_(econ,[0])])
            if(T_rep_prev>T_rep):
                process = 'cooling'
            phase = phase_determiner(T_rep=T_rep,process=process)
            rhos,cs,kappas = props_chooser((N@theta_prev_pic[np.ix_(econ,[0])])[0][0],phase=phase,process=process)
            rho_t,c_t,kappa_t = props_chooser((N@theta_prev_pic[np.ix_(econ,[0])])[0][0]+del_t,phase=phase,process=process)
            # kappa = N@kappas/1e3
            # rho = N@rhos/1e9 
            # c = N@cs
            kappa = kappas/1e3
            rho = rhos/1e9
            c = cs  
            kappa_t = kappa_t/1e3
            rho_t = rho_t/1e9

        del_k_del_t = (kappa_t - kappa)/del_t
        del_rho_del_t = (rho_t - rho)/del_t
        del_c_del_t = (c_t - c)/del_t
        K_loc += kappa*a
        K_D_loc += kappa*a + (a@theta_prev_pic[np.ix_(econ,[0])])*N*del_k_del_t
        G_loc += rho*c*vo*b
        M_loc += rho*c*m
        M_D_loc += rho*c*m + (m@theta_prev_pic[np.ix_(econ,[0])])*N*(del_c_del_t*rho+del_rho_del_t*c)
        # print("dc/dt " +str(del_c_del_t))
        # print("drho/dt " +str(del_rho_del_t))
        X = np.matmul(N,boundary)
        f_loc += N*Q(X,centre,ro)*np.linalg.det(Jac)*weights[k]
        area += np.linalg.det(Jac)*weights[k]  
    # M_D_loc = M_loc
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
            
            if K_D_loc[i][j]!=0:
                K_D_row.append(econ[i])
                K_D_col.append(econ[j])
                K_D_data.append(K_D_loc[i][j])

            if M_loc[i][j]!=0:
                M_row.append(econ[i])
                M_col.append(econ[j])
                M_data.append(M_loc[i][j])
                
            
            if M_D_loc[i][j]!=0:
                M_D_row.append(econ[i])
                M_D_col.append(econ[j])
                M_D_data.append(M_D_loc[i][j])

    for l,m in zip([0,1,2],[1,2,0]):
        n1 = econ[l]
        n2 = econ[m]
        check_rn = (n1 in rn and n2 in rn)
        check_tn = (n1 in tn and n2 in tn)
        check_bn = (n1 in bn and n2 in bn)
        check_ln = (n1 in ln and n2 in ln)

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
                
    return K_row, K_col, K_data, K_D_row, K_D_col, K_D_data ,G_row, G_col, G_data, M_row, M_col, M_data, M_D_row, M_D_col, M_D_data, F_row,F_data, BT_row, BT_data,area