import random
import numpy as np
import math

# reading text files to create a matrix
def readfile(f1):
    rows =[]
    # save the numbers as string in an array
    for line in f1:
        row = line.split()
        rows.append(row)
        # change str to float
    for i in range(len(rows)):
        for k in range(len(rows[0])):
            #print(a1[i][k])
            rows[i][k] = float(rows[i][k])
    return rows


# to read 1 D array from text file
def read1d(p):
    b = p.read()
    b1 = (b.split(' '))
    C=[]
    for i in range(len(b1)):
        C.append(0)
    for i in range(4):
        C[i] = float(b1[i])
    return C

# matrix multiplication
def matmul(M,A):
    B = []
    for i in range(len(M)):
        row = []
        for j in range(len(A[0])):
            row.append(0)
        B.append(row)

    for x in range(len(M)):
        for y in range(len(A[0])):
            for z in range(len(M[0])):
                B[x][y] += M[x][z] * A[z][y]
    return B

#forward substitution - LY= B. Find for Y. For linear Eqn
def fwrdsub(A , B):
    global Y
    Y = []
    for k in range(len(A)):
        Y.append(float(0))
    for i in range(0, len(A)):
        val = 0
        for j in range(0, i):
            val += A[i][j]*Y[j]
        Y[i] += (B[i] - val)
    return Y

# forward and backward substitution for cholesky method
def cholesky_FB(L):
    # L*L+*X = I and X is A inverse
    Y = [[0 for x in range(len(L))]
         for y in range(len(L))]
    n = len(L)
    # forward subtitution part
    for i in range(n):
        Y[i][i] = 1/L[i][i]
    for i in range(1, n):
        for j in range(i):
            sum = 0
            for k in range(j, i):
                sum += L[i][k]*Y[k][j]
            Y[i][j] = - sum / L[i][i]
    #Backward substitution part
    x = [[0 for i in range(n)] for j in range(n)]
    Lt = np.transpose(L)
    for i in reversed(range(n)):
        x[i][i] = (Y[i][i]) / (L[i][i])
    for i in reversed(range(n - 1)):
        for j in reversed(range((i + 1), n)):
            sum = 0
            for k in range(i + 1, j + 1):
                sum += ((Lt[i][k]) * (x[k][j]))

            x[i][j] = (Y[j][i] - sum) / (L[i][i])
    return x

#backward substitution- UX = Y find for X.
def bkwdsub(A ,B):
    global X
    X = []
    for k in range(len(A)):
        X.append(float(0))
    for i in reversed(range(len(A))):
        val = 0
        for j in reversed(range(0, len(A))):
            if j > i:
                val += A[i][j]*X[j]
        X[i] += (1/A[i][i])*(B[i] - val)
    return X

def LUdecomp (A, C):
    for j in range(len(A)):
        Parpivot(A, C, j)
        for i in range(len(A)):
            if i <= j:
                sumt = 0
                for k in range(0, i):
                        sumt += A[i][k] * A[k][j]
                A[i][j] = A[i][j] - sumt
            if i > j:
                sumt = 0
                for k in range(0, j):
                        sumt += A[i][k] * A[k][j]
                A[i][j] = (1/A[j][j])*(A[i][j]-sumt)
    return A



def GaussJordan(A, B):
    n = len(B)
    for k in range(n):
        Parpivot(A, B, k)
        # the pivot row
        pivot = A[k][k]
        # To divide entire pivot row by the pivot
        for i in range(k, n):
            A[k][i] = A[k][i]/pivot

        B[k] = B[k] / pivot
        # other rows
        for i in range(n):
            if abs(A[i][k]) < 1e-10 or i == k:
                continue
            else:
                term = A[i][k]
                for j in range(k, n):
                    A[i][j] = A[i][j] - term * A[k][j]
                B[i] = B[i] - term * B[k]
    return B, A

def Parpivot(A, B, k):
    if np.abs(A[k][k]) < 1e-10:
        n = len(B)
        for i in range(k + 1, n):
            if abs(A[i][k]) > abs(A[k][k]) and abs(A[k][k]) == 0:
                A[k], A[i] = A[i], A[k]
                B[k], B[i] = B[i], B[k]
    return A, B

# convert A to L*L+ where L is lower triangular
def cholesky(A):
    n = len(A)
    # create L matrix
    L = [[0 for x in range(n)]
         for y in range(n)]
    for i in range(len(A)):
        sum = 0
        # for diagonal elements
        for j in range(i):
            sum += L[i][j]**2
        L[i][i] = math.sqrt(A[i][i] - sum)
        # For non diagonal elements
        for k in range(i+1,n):
            sum = 0
            for j in range(i):
                sum += L[k][j]*L[i][j]
            L[k][i] = (1/L[i][i])*(A[k][i] - sum)
    return L

def cholesky_Inv(A):
    L = cholesky(A)
    Inv = cholesky_FB(L)
    return Inv


# Gauss Seidel to solve for linear eqn AX = B
def gauss_seidel(a,b,x,tolerance):
    a = np.array(a)
    x = np.array(x)
    #n = len(a)
    Iter = []
    Error = []
    #x = np.zeros_like(b)
    for k in range(100000):
        #another vector for updating x
        X = x.copy()
        for i in range(len(a[0])):
            sum1=0
            sum2=0
            x[i] = (b[i]- np.dot(a[i,:i],x[:i])-np.dot(a[i,(i+1):],X[(i+1):]))/a[i,i]
            err = np.linalg.norm(x - X,ord = np.inf)/np.linalg.norm(x, ord = np.inf)
            Error.append(err)
            Iter.append(k)

            if err<tolerance:
                break
        #     for j in range(i+1,len(a[0])):
        #         sum1 += (a[i][j]*X[j])
        #     for j in range(1,i):
        #         sum2 += (a[i][j]*x[j])
        #     X[i] = (b[i] - sum1 - sum2) / a[i][i]
        # if np.allclose(x, X, rtol=tolerance):
        #     break
        # x = X
    #error = np.dot(A, x) - B
    return x,Error, Iter


# finding inverse using gauss sidel
def gauss_sidel_Inv(a):
    n = len(a)
    b=np.identity(n)
    # calculating new x
    x = [0 for x in range(len(a[0]))]
    Inv=[]
    for k in range(n):
        for i in range(1, n+1):
            x_new = [0 for x in range(len(x))]
            #print("Iteration "+str(i)+" : "+str(x))
            for i in range(a.shape[0]):
                # finding terms
                s1 = np.dot(a[i, :i], x_new[:i])
                s2 = np.dot(a[i, i + 1 :], x[i + 1 :])
                x_new[i] = (b[k][i] - s1 - s2) / a[i, i]
            if np.allclose(x, x_new, rtol=1e-8):
                break
            x = x_new
        Inv.append(x)
    #print("Solution is: "+ str(x))
    error = np.dot(a, x)-b
    #print("Error is : "+ str(error))
    #This inverse is actually the transpose of the actual inverse
    return Inv, error



# Conjugate gradient
def conjugate_gradient(A,B,x,tolerance = 1e-4):
    A = np.array(A)
    B = np.array(B)
    x = np.array(x)
    # residual
    residue = []
    Iter = []
    r = B - A.dot(x)
    p = r.copy()
    # finding rT * r
    for i in range(10000):
        Ap = A.dot(p)
        if np.dot(p,Ap) == 0:
            break
        # alpha = rT*r/ (A*r)T * r
        alpha = np.dot(p, r)/(np.dot(p, Ap))
        # updating x and r
        x = x + (alpha * p)
        r = B - A.dot(x)

        # Residue and Iteration
        rs_new = np.sum(r**2)
        residue.append(np.sqrt(rs_new))
        Iter.append(i)

        if np.sqrt(rs_new) < tolerance:
            break
        else:
            c = -np.dot(r,Ap)/np.dot(r,Ap)
            p = r + (c* p)
    return x, residue, Iter

def inv_cal(A, method,x,tolerance):
    n = len(A)
    Inv = []
    Err = []
    Iter = []
    #res_list_comb = []
    for i in range(n):
        b = [0.0 for i in range(n)]
        b[i] = 1
        X = method(A, b, x,tolerance)
        Inv.append(X[0])
        #res_list_comb.extend(res_list)
    #print(inv)
    Err = X[1]
    Iter = X[2]
    Inv = np.transpose(np.array(Inv))
    return Inv, Err, Iter

# Inverse using Conjugate gradient
def conjugate_gradient_Inv(a):
    n = len(a)
    Inv=[]
    residue = []
    b = np.identity(n)
    for k in range(n):
        x = np.zeros_like(a[0])
        # residual
        r = b[k] - np.matmul(a,x)
        residue.append(r)
        p = r
        rs_old = np.matmul(np.transpose(r),r)
        for i in range(n):
            ap = np.matmul(a,p)
            alpha = rs_old/(np.matmul(np.transpose(p),ap))
            x += alpha*p
            r -= alpha*ap
            rs_new = np.matmul(np.transpose(r),r)
            if np.sqrt(rs_new) < 1e-12:
                break
            else:
                p = r + (rs_new/rs_old)* p
                rs_old = rs_new
        Inv.append(x)
        #This inverse is actually the transpose of the actual inverse
    return Inv





# Jacobi Method - All eigenvalues. For numpy array
def Jacobi_eval(A):
    # largest off-diag element
    n = len(A)
    def maxind(A):
        Amax = 0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j])>=Amax:
                    Amax = abs(A[i,j])
                    k = i
                    l = j
        return Amax, k,l

    # to make A[k,l] = 0 by rotation and define rotation matrix
    def rotate(A,p,k,l):
        A_diff = A[l,l]-A[k,k]
        if abs(A[k,l])< abs(A_diff)*1e-30:
            t = A[k,l]/A_diff
        else:
            phi = A_diff/(2*A[k,l])
            t = 1/(abs(phi)+np.sqrt(phi**2+1))
            if phi<0:
                t = -t
        c = 1/np.sqrt(t**2+1)
        s = t*c
        tau = s/(1+c)

        term = A[k,l]
        A[k,l] = 0
        A[k,k] = A[k,k] - t*term
        A[l,l] = A[l,l] + t*term
        for i in range(k):
            term = A[i,k]
            A[i,k] = term - s*(A[i,l] + tau*term )
            A[i,l] += s*(term- tau*A[i,l])
        for i in range(k+1,l):
            term = A[k,i]
            A[k,i] = term - s*(A[i,l] + tau*A[k,i])
            A[i,l] += s*(term - tau*A[i,l])
        for i in range(l+1, n):
            term = A[k,i]
            A[k,i] = term - s*(A[l,i] + tau*term)
            A[l,i] += s*(term - tau*A[l,i])
        for i in range(n):
            term = p[i,k]
            p[i,k] = term - s*(p[i,l] - tau*p[i,k])
            p[i,l] += s*(term - tau*p[i,l])

    p = np.identity(n)
    for i in range(4*n**2):
        Amax, k,l = maxind(A)
        if Amax < 1e-9:
            return np.diagonal(A)
        rotate(A,p,k,l)
    print("This method did not converge")


def chisquare(ob_bin,ex_bin,cnstrn = 1):
    df = len(ex_bin)-cnstrn
    chsq = 0
    for j in range(len(ex_bin)):
        if ex_bin[j]<= 0:
            print("Error in expected number of instances in bin")
        temp = ob_bin[j]-ex_bin[j]
        chsq += temp*temp/ex_bin[j]
    return chsq

def chisqtwo(bin1,bin2,cnstrn=1):
    df = len(bin1) - cnstrn
    chsq = 0
    for j in range(len(bin1)):
        if bin1[j] == 0 and bin2[j] == 0:
            df -= 1
        else:
            chsq += pow((bin1[j]-bin2[j]),2)/(bin1[j]+bin2[j])


def jacobi(a, b, x,tolerance):
    a = np.array(a)
    b = np.array(b)
    # iteration
    Itr = 6000
    Iter =[]
    Error =[]
    # x is the initial guess of x
    # k is the iteration limit
    n = len(b)
    for z in range(Itr):
        Iter.append(z)

        X = np.array([0.0 for i in range(n)])
        for i in range(n):
            sum = 0
            for j in range(n):
                if i != j:
                    sum += ((a[i][j]) * x[j])
            X[i] = (1 / (a[i][i])) * (b[i] - sum)
            if X[i] == X[i-1]:
                break
        err = 0
        for i in X:
            err+=i**2
        err = np.sqrt(err)
        Error.append(err)
        # if np.allclose(x, X, atol=tolerance):
        #     break
        x = X
    return X,Error,Iter


def jacobi_Inv(a):
    # initial guess x
    # iteration limit taken as 15
    n = len(a)
    b = np.identity(len(a))
    X = np.array([0.0 for i in range(n)])
    inverse = []
    k = 15
    for m in range(n):
        x = [1 for i in range(n)]
        for z in range(k):
            for i in range(n):
                sum = 0
                for j in range(n):
                    if i != j:
                        sum += ((a[i][j]) * x[j])
                X[i] = (1 / (a[i][i])) * (b[m][i] - sum)
            x = X
        inverse.append(x)
    # This inverse is actually the transpose of the actual inverse
    return inverse



def Linear_reg(x, y, sigma):
    a = 0
    b = 0
    chisq = 0
    n = len(x)
    S = 0;Sx = 0;Sy = 0; Sxx = 0;Sxy = 0
    global cov_ab; global errora; global errorb
    y_p = [1 for x in range(n)]
    for i in range(len(x)):
        y_p[i] = a + b*x[i]
        chisq += ((y[i] - y_p[i])/sigma[i])**2
        if chisq <= 1e-10:
            break
        #dchia += -2*(y[i] -y_p[i])/sigma[i]**2
        #dchib += -2*x[i]*(y[i]-y[p])/sigma[i]**2
        S += 1/sigma[i]**2
        Sx += x[i]/sigma[i]**2
        Sy += y[i]/sigma[i]**2
        Sxx += (x[i]/sigma[i])**2
        Sxy += x[i]*y[i]/sigma[i]**2

    delta = S*Sxx-(Sx)**2
    a = (Sxx*Sy-Sx*Sxy)/delta
    b = (S*Sxy-Sx*Sy)/delta

    cov_ab = -Sx/delta
    errora = np.sqrt(Sxx/delta)
    errorb = np.sqrt(S/delta)
    return a,b, cov_ab, errora, errorb



def power_method(A,x,y,n):
    #x is the initial guess\
    #x=c1v1+c2v2+c3v3+........
    #Assume a simple possible dominating eigenvector with norm equal to 1
    #y be any vector not orthogonal to v1
    x2 = x
    import numpy as np
    #n is number of iteration
    for i in range(n):
        x = matmul(A, x)
        norm = np.linalg.norm(x)
        #normalizing u1
        X1 = x/norm
    for i in range(n-1):
        x2 = matmul(A, x2)
        norm = np.linalg.norm(x2)
        #normalizing u1
        X2 = x2/norm
    #calculating the approximate dominating eigenvalue
    eigvalue = np.dot(x,y)/np.dot(x2,y)
    #
    #calculation of approximate dominating eigenvector
    eigvect = X2
    return eigvalue, eigvect

def power_method_ND(A,x,y,n,m):
    #m is the number of non-dominant eigen vectors needed
    eigval=[]
    eigvec=[]
    for i in range(m):
        eigvalue, eigvect = power_method(A, x, y, n)
        A = A - (eigvalue*(matmul(eigvect, (np.transpose(eigvect)))))
        eigval.append(eigvalue)
        eigvec.append(eigvect)
    return eigval,eigvec


def Jackknife(a):
    # you should convert the given dataset into a matrix
    # and then evaluate it's number of columns to know in
    # how many rows we need to apply the processes of mean and all
    m = len(a)
    n = len(a[0])
    # n is the number of elements in each column
    # m is the number of columns(number of 1D arrays inside the 2D array)
    y_meank = []
    y_mean_sqk = []
    # rows and columns interchanged to make code easier
    for i in range(m):
        yk_each = []
        yk_each_sq = []
        for k in range(n):
            y_sum = 0
            for j in range(n):
                if j != k:
                    y_sum += a[i][j]

            yk_each.append(y_sum / (n - 1))
            yk_each_sq.append((y_sum / (n - 1)) ** 2)
        y_meank.append(yk_each)
        y_mean_sqk.append(yk_each_sq)
    y_mean_jk = []
    y_mean_sq_kmean = []
    for i in range(m):
        y_sum = 0
        y_sum_square = 0
        for j in range(n):
            y_sum += y_meank[i][j]
            y_sum_square += y_mean_sqk[i][j]
        y_mean_jk.append(y_sum / n)
        y_mean_sq_kmean.append(y_sum_square / n)
        # here each element of y_mean_jk represent mean of
        # each column(where each element of the column is y_k bar)
    sigma_jk_sq = np.subtract(y_mean_sq_kmean, matmul(y_mean_jk, y_mean_jk))
    sigma = (n - 1) * (sigma_jk_sq)
    # here sigma and sigma_jk_square are the variances
    return y_mean_jk, sigma_jk_sq, sigma

from scipy import linalg
def poly_fit(x, y, Dy, order=1):
    # takes in x y and Dy array and order of poly to be fitted.
    # make vector Y -> make matrix X -> take invers matrix X -> multiply X with Y
    X = [];Y = [];B = []
    para = order + 1  # no of parameters
    ar_x = np.array(x)
    ar_y = np.array(y)
    ar_Dy = np.array(Dy)
    for i in range(para):
        X.append([])
        for j in range(para):
            X[i].append(np.sum(((ar_x ** (i + j))) / (ar_Dy ** 2)))

    for i in range(para):
        B.append([])
        for j in range(para):
            if (i == j):
                B[i].append(1)
            else:
                B[i].append(0)
    for i in range(para):
        Y.append([])
        Y[i].append(np.sum(np.multiply((ar_x ** i), ar_y) / ar_Dy ** 2))

    inv = linalg.inv(np.array(X))
    parameter = np.dot(inv, Y)
    return parameter

import matplotlib.pyplot as plt

def plot(x,y,sigma):
    a,b,cov_ab, errora, errorb = Linear_reg(x,y,sigma)
    x = np.linspace(0,100,1000)
    plt.plot(x, a + b*x)
    plt.show()
