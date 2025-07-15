import numpy as np

def Q(point, centre, qo, ro):
    x = point[0,0] - centre[0,0]
    y = point[0,1] - centre[0,1]
    return qo*np.exp(-(x**2+y**2)/ro**2)  ## W/mm^3

#we consider only one phase - alpha
def rho_Ti(T):
    return -5.13e-5*(T**2)-0.01935*T+4451

def cp_Ti(T, process='heating'):
    lin = 0.25*T+483
    heat = 13000*np.exp(-0.5*(((T-1160)/90)**2))/(90*np.sqrt(2*np.pi))
    cool = 13000*np.exp(-0.5*(((T-952)/90)**2))/(90*np.sqrt(2*np.pi))
    if process == 'heating':
        return lin+heat
    elif process == 'cooling':
        return lin+cool

def k_Ti(T):
    return 0.012*T+3.3

def props_chooser(T, process):
    return rho_Ti(T), cp_Ti(T, process), k_Ti(T)

def nr_helper_rect(args):
    nodes, ele, centre, theta_prev_time, theta_prev2_time, theta_prev_nr = args
    gp = 3
    
    M_row,M_col,M_data = [],[],[]
    K_row,K_col,K_data = [],[],[]
    G_row,G_col,G_data = [],[],[]
    dMT_row,dMT_col,dMT_data = [],[],[]
    dKT_row,dKT_col,dKT_data = [],[],[]
    dGT_row,dGT_col,dGT_data = [],[],[]
    F_row,F_data = [],[]
    BT_row,BT_data = [],[]
    
    ro = 0.75 #mm
    qo = 440/(np.pi*ro*ro)   # W/mm^2
    cp = 658 #J/kg.K
    rho = 7.6e-6 #kg/mm^3
    kappa = 0.025 #W/mm.K
    vo = 10 #mm/s

    dkappa = 0
    drho = 0
    dcp = 0
    
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

    #determine whether the element is being heated or not
    T_rep = np.mean(theta_prev_time[np.ix_(econ,[0])]) #temperature at the centroid of the element
    if theta_prev2_time is None:
        process = 'heating'
    else:
        T_rep_prev = np.mean(theta_prev2_time[np.ix_(econ,[0])]) 
        if T_rep>=T_rep_prev:
            process = 'heating'
        else:
            process = 'cooling'

    M_loc = np.zeros((nnode,nnode))
    K_loc = np.zeros((nnode,nnode))
    G_loc = np.zeros((nnode,nnode))
    dMT_loc = np.zeros((nnode,nnode))
    dKT_loc = np.zeros((nnode,nnode))
    dGT_loc = np.zeros((nnode,nnode))
    f_loc = np.zeros((nnode,nnode))

    delta = 1e-3
    h = 10 #to be reconsidered
    #nodes already reordered
    x1, y1 = nodes[econ[0],:2] 
    x2, y2 = nodes[econ[1],:2] 
    x3, y3 = nodes[econ[2],:2]
    d12 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    d23 = np.sqrt((x3-x2)**2+(y3-y2)**2)
    d31 = np.sqrt((x1-x3)**2+(y1-y3)**2)
    d = np.array([[0, d12, d31], [d12, 0, d23], [d31, d23, 0]])

    #convection boundary term
    ln = np.where(nodes[:,0] == 0)[0]
    rn = np.where(nodes[:,0] == np.max(nodes[:,0]))[0]
    bn = np.where(nodes[:,1] == 0)[0]
    tn = np.where(nodes[:,1] == np.max(nodes[:,1]))[0]

    T_inf = 273+27 #to be reconsidered

    for l,m in zip([0,1,2],[1,2,0]):
        n1 = econ[l]
        n2 = econ[m]
        d12 = d[l, m]
        check_ln = (n1 in ln and n2 in ln)
        check_rn = (n1 in rn and n2 in rn)
        check_tn = (n1 in tn and n2 in tn)
        check_bn = (n1 in bn and n2 in bn)

        if check_rn or check_tn or check_bn or check_ln:
            BT_row.append(n1)
            BT_row.append(n2)
            BT_data.append(h*T_inf*d12/2)
            BT_data.append(h*T_inf*d12/2)
            K_loc[l, l], K_loc[m, m], K_loc[l, m], K_loc[m, l] = d12/3, d12/3, d12/6, d12/6
            dKT_loc[l, l], dKT_loc[m, m], dKT_loc[l, m], dKT_loc[m, l] = d12/3, d12/3, d12/6, d12/6

    for k,ipk in enumerate(ips):
        N = np.array([[(1-ipk[0]-ipk[1]), ipk[0],ipk[1]]])
        a = (Jac_inv@dN).T@(Jac_inv@dN)*(np.linalg.det(Jac))*weights[k]
        b = (N.T@N)*(np.linalg.det(Jac))*weights[k]
        c = N.T@dN_dx*(np.linalg.det(Jac))*weights[k]


        rhos,cps,kappas = props_chooser(theta_prev_nr[np.ix_(econ,[0])], process)
        rhos_n,cps_n,kappas_n = props_chooser(theta_prev_nr[np.ix_(econ,[0])]+delta, process)
        kappa = N@kappas/1e3
        rho = N@rhos/1e9 
        cp = N@cps
        kappa_n = N@kappas_n/1e3
        rho_n = N@rhos_n/1e9 
        cp_n = N@cps_n
        dkappa = (kappa_n-kappa)/delta
        drho = (rho_n-rho)/delta
        dcp = (cp_n-cp)/delta

        K_loc += kappa*a
        M_loc += rho*cp*b
        G_loc += rho*cp*vo*c
        dKT_loc += kappa*a + dkappa*(a@theta_prev_nr[np.ix_(econ,[0])])@N
        dMT_loc += rho*cp*b +(drho*cp+dcp*drho)*(b@theta_prev_nr[np.ix_(econ,[0])])@N
        dGT_loc += rho*cp*vo*c + vo*(drho*cp+dcp*drho)*(c@theta_prev_nr[np.ix_(econ,[0])])@N
        X  =np.matmul(N,boundary)
        f_loc += N*Q(X, centre, qo, ro)*np.linalg.det(Jac)*weights[k]
        area += np.linalg.det(Jac)*weights[k]  

    for i in range(nnode):
        if (f_loc.T[i,0]):
            F_row.append(econ[i])
            F_data.append(f_loc.T[i,0])
        for j in range(nnode):
            if M_loc[i][j]!=0:
                M_row.append(econ[i])
                M_col.append(econ[j])
                M_data.append(M_loc[i][j])

            if K_loc[i][j]!=0:
                K_row.append(econ[i])
                K_col.append(econ[j])
                K_data.append(K_loc[i][j])
                
            if G_loc[i][j]!=0:
                G_row.append(econ[i])
                G_col.append(econ[j])
                G_data.append(G_loc[i][j])
            
            if dMT_loc[i][j]!=0:
                dMT_row.append(econ[i])
                dMT_col.append(econ[j])
                dMT_data.append(dMT_loc[i][j])

            if dKT_loc[i][j]!=0:
                dKT_row.append(econ[i])
                dKT_col.append(econ[j])
                dKT_data.append(dKT_loc[i][j])
                
            if dGT_loc[i][j]!=0:
                dGT_row.append(econ[i])
                dGT_col.append(econ[j])
                dGT_data.append(dGT_loc[i][j])

    return M_row, M_col, M_data,K_row, K_col, K_data, G_row, G_col, G_data, \
           dMT_row, dMT_col, dMT_data,dKT_row, dKT_col, dKT_data, dGT_row, dGT_col, dGT_data,\
           F_row,F_data, BT_row, BT_data,area
   
def picard_helper_rect(args):
    nodes,ele,centre,theta_prev_time,theta_prev_nr,mode = args
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
            kappa = N@k_T(theta_prev_nr[np.ix_(econ,[0])])
            rho = N@rho_T(theta_prev_nr[np.ix_(econ,[0])])  
            c = N@cp_T(theta_prev_nr[np.ix_(econ,[0])])
            
        elif mode == "phase_change":
            rhos,cs,kappas = props_chooser(theta_prev_nr[np.ix_(econ,[0])],T_rep)
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