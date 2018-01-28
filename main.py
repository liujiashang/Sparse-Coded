#!/usr/bin/env python

from mpi4py import MPI
from scipy.sparse import csr_matrix, vstack, hstack
import scipy.io
import numpy as np
import random
import time

# Matrix division
m = 3
n = 3
k = m * n
N = 9
NRA = 1200000
NCA = 1200000
NCB = 1200000

NR = NRA // m
NC = NCB // n

comm = MPI.COMM_WORLD

if comm.rank == 0:
    # Master
    print("Running with %d processes:" % comm.Get_size())

    # Read matrix from files and construct the matrices

    Matrix_A = scipy.io.loadmat('A.mat')
    Matrix_A = csr_matrix(Matrix_A['A'])
    Matrix_A = Matrix_A.tocoo()

    A_row = np.array(Matrix_A.row, dtype=int)
    A_col = np.array(Matrix_A.col, dtype=int)
    A_data = np.array(Matrix_A.data, dtype=float)
    As = []
    Ad = []
    Ar = []
    Ac = []

    for i in range(m):
        left = i * NR
        right = (i + 1) * NR - 1
        tmp_idx_A = np.where(abs(A_row - right) + abs(A_row - left) <= NR)
        idx_A = tmp_idx_A[0]
        Ar.append(np.array(A_row[idx_A] - left, dtype=int))
        Ac.append(np.array(A_col[idx_A], dtype=int))
        Ad.append(np.array(A_data[idx_A], dtype=float))
        As.append(np.array([idx_A.size], dtype=int))

    B_row = A_row
    B_col = A_col
    B_data = A_data
    Bs = []
    Bd = []
    Br = []
    Bc = []

    for i in range(n):
        up = i * NC
        down = (i + 1) * NC - 1
        tmp_idx_B = np.where(abs(B_col - up) + abs(B_col - down) <= NC)
        idx_B = tmp_idx_B[0]
        Br.append(np.array(B_row[idx_B], dtype=int))
        Bc.append(np.array(B_col[idx_B] - up, dtype=int))
        Bd.append(np.array(B_data[idx_B], dtype=float))
        Bs.append(np.array([idx_B.size], dtype=int))

    # Input Matrix M
    M = np.loadtxt("M.txt")

    # Transmit the matrices to workers
    tm = [None] * N
    tAs = [None] * N
    tAd = [None] * N
    tAr = [None] * N
    tAc = [None] * N
    tBs = [None] * N
    tBd = [None] * N
    tBr = [None] * N
    tBc = [None] * N

    start_transmission_time = time.time()

    for i in range(N):
        tm[i] = comm.Isend(np.array(M[i], dtype=int), dest=i + 1, tag=5)
        tot = np.int(sum(M[i]))
        tAs[i] = [None] * tot
        tAd[i] = [None] * tot
        tAr[i] = [None] * tot
        tAc[i] = [None] * tot
        tBs[i] = [None] * tot
        tBd[i] = [None] * tot
        tBr[i] = [None] * tot
        tBc[i] = [None] * tot
        now = 0
        for idx in range(m*n):
            if np.int(M[i][idx]) == 1:
                tAs[i][now] = comm.Isend(As[idx // n], dest=i + 1, tag=100+now)
                tAd[i][now] = comm.Isend(Ad[idx // n], dest=i + 1, tag=200+now)
                tAr[i][now] = comm.Isend(Ar[idx // n], dest=i + 1, tag=300+now)
                tAc[i][now] = comm.Isend(Ac[idx // n], dest=i + 1, tag=400+now)

                tBs[i][now] = comm.Isend(Bs[idx % n], dest=i + 1, tag=500+now)
                tBd[i][now] = comm.Isend(Bd[idx % n], dest=i + 1, tag=600+now)
                tBr[i][now] = comm.Isend(Br[idx % n], dest=i + 1, tag=700+now)
                tBc[i][now] = comm.Isend(Bc[idx % n], dest=i + 1, tag=800+now)
                now = now+1

    for i in range(N):
        MPI.Request.Waitall(tAd[i])
        MPI.Request.Waitall(tAr[i])
        MPI.Request.Waitall(tAc[i])
        MPI.Request.Waitall(tBd[i])
        MPI.Request.Waitall(tBr[i])
        MPI.Request.Waitall(tBc[i])

    finish_transmission_time = time.time()
    print ("Transmitted, %f" % (finish_transmission_time-start_transmission_time))

    # Receive results from workers

    transmit_finished = time.time()

    rPs = [None] * N
    partial_result_size = []

    for i in range(N):
        partial_result_size.append(np.array([0] * 1, dtype=int))

    for i in range(N):
        rPs[i] = comm.Irecv(partial_result_size[i], source=i+1, tag=70)

    R_worker = []
    R_M = np.zeros((N, m*n), dtype=int)
    R_Partial_Result_data = []
    R_Partial_Result_row = []
    R_Partial_Result_col = []

    Receive_length = 0
    R_sig_data = []
    R_sig_row = []
    R_sig_col = []

    while np.linalg.matrix_rank(R_M) < m*n:
        j = MPI.Request.Waitany(rPs)
        R_Partial_Result_data.append(np.array([0] * partial_result_size[j][0], dtype=float))
        R_Partial_Result_row.append(np.array([0] * partial_result_size[j][0], dtype=int))
        R_Partial_Result_col.append(np.array([0] * partial_result_size[j][0], dtype=int))

        R_sig_data.append(comm.Irecv(R_Partial_Result_data[Receive_length], source=j+1, tag=75))
        R_sig_row.append(comm.Irecv(R_Partial_Result_row[Receive_length], source=j+1, tag=80))
        R_sig_col.append(comm.Irecv(R_Partial_Result_col[Receive_length], source=j+1, tag=85))
        R_worker.append(j)
        R_M[Receive_length] = M[j]
        Receive_length = Receive_length + 1

    MPI.Request.Waitall(R_sig_data)
    MPI.Request.Waitall(R_sig_col)
    MPI.Request.Waitall(R_sig_row)

    received_finished = time.time()

    print ("Processing time %f" % (received_finished-transmit_finished))
    print (R_worker)

    Partial_Result = [None] * Receive_length

    for i in range(Receive_length):
        Partial_Result[i] = csr_matrix((R_Partial_Result_data[i], (R_Partial_Result_row[i], R_Partial_Result_col[i])),
                                       (NR, NC))

    R_M = R_M[:Receive_length]

    decode_start = time.time()

    # Decode the results
    C = [None] * (m*n)
    Recovered = [0] * (m*n)

    already_done = 0

    borrow = 0

    while already_done < m*n:
        # find ripple
        ripple = []
        for i in range(Receive_length):
            if sum(R_M[i]) == 1:
                tmp_idx_M = np.where(R_M[i])
                idx = tmp_idx_M[0][0]
                if Recovered[idx] == 0:
                    Recovered[idx] = 1
                    ripple.append(idx)
                    C[idx] = Partial_Result[i]
                    already_done = already_done + 1

        while (already_done < m*n) and (len(ripple) > 0):
            next = []
            for rip in ripple:
                for row in range(Receive_length):
                    if R_M[row][rip] == 1:
                        R_M[row][rip] = 0
                        Partial_Result[row] = Partial_Result[row] - C[rip]
                        if sum(R_M[row]) == 1:
                            tmp_idx_M = np.where(R_M[row])
                            idx = tmp_idx_M[0][0]
                            if Recovered[idx] == 0:
                                Recovered[idx] = 1
                                next.append(idx)
                                C[idx] = Partial_Result[row]
                                already_done = already_done + 1
            ripple = next

        if already_done == m*n:
            break

        borrow = borrow + 1

        idx_sum = np.sum(R_M, axis=0)
        next_recover = idx_sum.argmax()

        u = np.zeros((m*n, 1), dtype=int)
        u[next_recover] = 1

        M_Transpose = R_M.T

        valid_row = np.where(np.sum(M_Transpose, axis=1))

        u = u[valid_row]
        M_Transpose = M_Transpose[valid_row]

        chosen_x = np.linalg.lstsq(M_Transpose, u, rcond=10**-5)
        chosen_x = chosen_x[0]

        C[next_recover] = csr_matrix((NR, NC))

        for i in range(Receive_length):
            C[next_recover] += chosen_x[i][0] * Partial_Result[i]

        already_done = already_done + 1
        Recovered[next_recover] = 1

        for row in range(Receive_length):
            if R_M[row][next_recover] == 1:
                R_M[row][next_recover] = 0
                Partial_Result[row] = Partial_Result[row] - C[next_recover]

    print ("borrow: %d" % borrow)

    decode_finished = time.time()

    print ("Decoding time %f" % (decode_finished - decode_start))

    # for i in range(m):
    #     start_idx = i*n
    #     for j in range(1, n):
    #         C[start_idx] = hstack([C[start_idx], C[start_idx+j]])
    #
    # for i in range(1, m):
    #     idx = i*n
    #     C[0] = vstack([C[0], C[idx]])
    #
    # Matrix_A = scipy.io.loadmat('A.mat')
    # Matrix_A = csr_matrix(Matrix_A['A'])
    #
    # Correct = np.dot(Matrix_A, Matrix_A)
    #
    # print (Correct-C[0] > 10**-10)

else:
    # receive the number of A and B matrices needed to compute

    rMatrix = np.array([0] * m * n)
    rM = comm.Irecv(rMatrix, source=0, tag=5)

    rM.wait()

    NoM = np.int(sum(rMatrix))

    # receive matrix size

    R_A_size = []
    R_B_size = []

    for i in range(NoM):
        R_A_size.append(np.array([0] * 1, dtype=int))
        R_B_size.append(np.array([0] * 1, dtype=int))

    for cur in range(NoM):
        comm.Recv(R_A_size[cur], source=0, tag=100+cur)
        comm.Recv(R_B_size[cur], source=0, tag=500+cur)

    R_A_data = []
    R_A_row = []
    R_A_col = []

    R_B_data = []
    R_B_row = []
    R_B_col = []

    for cur in range(NoM):
        R_A_data.append(np.array([0]*R_A_size[cur][0], dtype=float))
        R_A_row.append(np.array([0]*R_A_size[cur][0], dtype=int))
        R_A_col.append(np.array([0]*R_A_size[cur][0], dtype=int))
        R_B_data.append(np.array([0]*R_B_size[cur][0], dtype=float))
        R_B_row.append(np.array([0]*R_B_size[cur][0], dtype=int))
        R_B_col.append(np.array([0]*R_B_size[cur][0], dtype=int))

    rAd = [None] * NoM
    rAr = [None] * NoM
    rAc = [None] * NoM
    rBd = [None] * NoM
    rBr = [None] * NoM
    rBc = [None] * NoM

    # receive matrix

    for cur in range(NoM):
        rAd[cur] = comm.Irecv(R_A_data[cur], source=0, tag=200+cur)
        rAr[cur] = comm.Irecv(R_A_row[cur], source=0, tag=300+cur)
        rAc[cur] = comm.Irecv(R_A_col[cur], source=0, tag=400+cur)
        rBd[cur] = comm.Irecv(R_B_data[cur], source=0, tag=600+cur)
        rBr[cur] = comm.Irecv(R_B_row[cur], source=0, tag=700+cur)
        rBc[cur] = comm.Irecv(R_B_col[cur], source=0, tag=800+cur)

    for cur in range(NoM):
        rAd[cur].wait()
        rAr[cur].wait()
        rAc[cur].wait()
        rBd[cur].wait()
        rBr[cur].wait()
        rBc[cur].wait()

    # compute result

    start = time.time()

    result = csr_matrix((NR, NC), dtype=float)

    for cur in range(NoM):
        result += np.dot(csr_matrix((R_A_data[cur], (R_A_row[cur], R_A_col[cur])), (NR, NCA)),
                         csr_matrix((R_B_data[cur], (R_B_row[cur], R_B_col[cur])), (NCA, NC)))

    result = result.tocoo()

    finished = time.time()

    # send result back to Master

    s = np.array([result.size], dtype=int)
    d = np.array(result.data, dtype=float)
    r = np.array(result.row, dtype=int)
    c = np.array(result.col, dtype=int)

    ts = comm.Isend(s, dest=0, tag=70)
    td = comm.Isend(d, dest=0, tag=75)
    tr = comm.Isend(r, dest=0, tag=80)
    tc = comm.Isend(c, dest=0, tag=85)

    ts.wait()
    td.wait()
    tr.wait()
    tc.wait()

    print("worker %d, %f, %f, %f, %d" % (comm.rank - 1, (finished - start), (time.time() - finished),
                                         (time.time() - start), s[0]))
