import numpy as np

def apply_bc(n1, n2, d12, l, m, bc, BT_row, BT_data, K_loc, dKT_loc):
    if bc["mode"] == "convection":
        h = bc["h"]
        T_inf = bc["T_inf"]
        BT_row.append(n1)
        BT_row.append(n2)
        BT_data.append(h*T_inf*d12/2)
        BT_data.append(h*T_inf*d12/2)
        K_loc[l, l], K_loc[m, m], K_loc[l, m], K_loc[m,l] = d12/3, d12/3, d12/6, d12/6
        dKT_loc[l, l], dKT_loc[m, m], dKT_loc[l,m], dKT_loc[m, l] = d12/3, d12/3, d12/6, d12/6
    elif bc["mode"] == "const_T":
        #we technically ignore these and enforce them separately
        pass
    elif bc["mode"] == "const_flux":
        q_ext = bc["q_ext"] #negative sign if inward
        if q_ext:
            BT_row.append(n1)
            BT_row.append(n2)
            BT_data.append(q_ext*d12/2)
            BT_data.append(q_ext*d12/2)
    else:
        raise Exception("No boundary condition defined as such")
    
def nr_helper(args):
    (problem_params, nodes, ele, source, theta_prev_time, theta_prev2_time, theta_prev_nr, props_chooser, boundary_conditions) = args
    gp = 3

    M_row, M_col, M_data = [], [], []
    K_row, K_col, K_data = [], [], []
    dMT_row, dMT_col, dMT_data = [], [], []
    dKT_row, dKT_col, dKT_data = [], [], []
    F_row, F_data = [], []
    BT_row, BT_data = [], []

    ro = problem_params["ro"]  # mm

    dkappa = 0
    drho = 0
    dcp = 0

    data_tle = {"ips": {1: [[1/3, 1/3]], 3: [[1/6, 1/6], [1/6, 2/3], [2/3, 1/6]]},
                "weights": {1: [1/2], 3: [1/6, 1/6, 1/6]}}

    ips = np.array(data_tle["ips"][gp])
    weights = np.array(data_tle["weights"][gp])
    econ = ele-1
    nnode = econ.shape[0]
    boundary = nodes[np.ix_(econ, [0, 1])]
    dN = np.array([[-1, 1, 0], [-1, 0, 1]])
    Jac = np.matmul(dN, boundary)
    if np.linalg.det(Jac) < 0:
        # reordering for the direction to be counter clockwise
        econ[0], econ[1] = econ[1], econ[0]
        boundary = nodes[np.ix_(econ, [0, 1])]
        Jac = np.matmul(dN, boundary)

    Jac_inv = np.linalg.inv(Jac)

    area = 0

    # determine whether the element is being heated or not
    # for non transient cases, it will always correspond to heating
    T_rep = np.mean(theta_prev_time[np.ix_(econ, [0])]) # temperature at the centroid of the element
    if theta_prev2_time is None:
        process = 'heating'
    else:
        T_rep_prev = np.mean(theta_prev2_time[np.ix_(econ, [0])])
        if T_rep >= T_rep_prev:
            process = 'heating'
        else:
            process = 'cooling'

    M_loc = np.zeros((nnode, nnode))
    K_loc = np.zeros((nnode, nnode))
    dMT_loc = np.zeros((nnode, nnode))
    dKT_loc = np.zeros((nnode, nnode))
    f_loc = np.zeros((nnode, nnode))

    # nodes already reordered
    x1, y1 = nodes[econ[0], :2]
    x2, y2 = nodes[econ[1], :2]
    x3, y3 = nodes[econ[2], :2]
    d12 = np.sqrt((x2-x1)**2+(y2-y1)**2)
    d23 = np.sqrt((x3-x2)**2+(y3-y2)**2)
    d31 = np.sqrt((x1-x3)**2+(y1-y3)**2)
    d = np.array([[0, d12, d31], [d12, 0, d23], [d31, d23, 0]])

    # convection boundary term
    ln = np.where(nodes[:, 0] == 0)[0]
    rn = np.where(nodes[:, 0] == np.max(nodes[:, 0]))[0]
    bn = np.where(nodes[:, 1] == 0)[0]
    tn = np.where(nodes[:, 1] == np.max(nodes[:, 1]))[0]


    for l, m in zip([0, 1, 2], [1, 2, 0]):
        n1 = econ[l]
        n2 = econ[m]
        d12 = d[l, m]
        check_ln = (n1 in ln and n2 in ln)
        check_rn = (n1 in rn and n2 in rn)
        check_tn = (n1 in tn and n2 in tn)
        check_bn = (n1 in bn and n2 in bn)

        if check_rn:
            apply_bc(n1, n2, d12, l, m, boundary_conditions["right"], BT_row, BT_data, K_loc, dKT_loc)
        elif check_bn:
            apply_bc(n1, n2, d12, l, m, boundary_conditions["bottom"], BT_row, BT_data, K_loc, dKT_loc)
        elif check_ln:
            apply_bc(n1, n2, d12, l, m, boundary_conditions["left"], BT_row, BT_data, K_loc, dKT_loc)
        elif check_tn:
            apply_bc(n1, n2, d12, l, m, boundary_conditions["top"], BT_row, BT_data, K_loc, dKT_loc)
        else:
            continue

    delta = 1e-3
    for k, ipk in enumerate(ips):
        N = np.array([[(1-ipk[0]-ipk[1]), ipk[0], ipk[1]]])
        a = (Jac_inv@dN).T@(Jac_inv@dN)*(np.linalg.det(Jac))*weights[k]
        b = (N.T@N)*(np.linalg.det(Jac))*weights[k]

        rhos, cps, kappas = props_chooser(
            theta_prev_nr[np.ix_(econ, [0])], process)
        rhos_n, cps_n, kappas_n = props_chooser(
            theta_prev_nr[np.ix_(econ, [0])]+delta, process)
        
        kappa = N@kappas
        rho = N@rhos
        cp = N@cps
        kappa_n = N@kappas_n
        rho_n = N@rhos_n
        cp_n = N@cps_n
        dkappa = (kappa_n-kappa)/delta
        drho = (rho_n-rho)/delta
        dcp = (cp_n-cp)/delta

        K_loc += kappa*a
        M_loc += rho*cp*b
        dKT_loc += kappa*a + dkappa*(a@theta_prev_nr[np.ix_(econ, [0])])@N
        dMT_loc += rho*cp*b + (drho*cp+dcp*drho) * \
            (b@theta_prev_nr[np.ix_(econ, [0])])@N
        X = np.matmul(N, boundary)
        f_loc += N*problem_params["Q"](X, source, ro)*np.linalg.det(Jac)*weights[k]
        area += np.linalg.det(Jac)*weights[k]

    for i in range(nnode):
        if (f_loc.T[i, 0]):
            F_row.append(econ[i])
            F_data.append(f_loc.T[i, 0])
        for j in range(nnode):
            if M_loc[i][j] != 0:
                M_row.append(econ[i])
                M_col.append(econ[j])
                M_data.append(M_loc[i][j])

            if K_loc[i][j] != 0:
                K_row.append(econ[i])
                K_col.append(econ[j])
                K_data.append(K_loc[i][j])

            if dMT_loc[i][j] != 0:
                dMT_row.append(econ[i])
                dMT_col.append(econ[j])
                dMT_data.append(dMT_loc[i][j])

            if dKT_loc[i][j] != 0:
                dKT_row.append(econ[i])
                dKT_col.append(econ[j])
                dKT_data.append(dKT_loc[i][j])

    return M_row, M_col, M_data, K_row, K_col, K_data, dMT_row, dMT_col, dMT_data, \
           dKT_row, dKT_col, dKT_data, F_row, F_data, BT_row, BT_data, area

