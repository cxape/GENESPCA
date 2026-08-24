import numpy as np
import scipy
import logging
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.gaussian_process.kernels import Matern
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import eigh
from scipy.optimize import Bounds
import time
from .utils_manopt import manopt_minimize
from scipy.optimize import brentq
from scipy.spatial import distance_matrix
logger = logging.getLogger(__name__)


class SMOPCA_relax:
    def __init__(self, Y_list, pos1, pos2, kernel="Matern", Z_dim=20, omics_weight=False, 
                 split=20, alpha_list=None, estimate_length=True,same_length=False, N01=False,sparse=False,cut_off=1e-20,KNN=False,KNN_n=5):
        """
        :param Y_list: data matrices from different modalities with shape (#feats, #cells)
        :param pos: spatial coordinates with shape (#cells, 2)
        :param Z_dim: dimension of latent factors
        :param omics_weight: choose if using weighted posterior for different modalities
        :param alpha_list: numpy array, weights of different modalities
        :param intercept: whether to use intercept for data with mean structures
        :param kernel_type: type of kernel, default is matern
        :param nu: matern kernel parameter, common value is 0.5, 1.5 or 2.5
        """
        assert all(Y.shape[1] == Y_list[0].shape[1] for Y in Y_list)
        self.Y_list = Y_list
        self.m_list = [Y.shape[0] for Y in Y_list]
        self.n = Y_list[0].shape[1]
        self.d = Z_dim
        self.modality_num = len(Y_list)

        self.kernel = kernel
        self.estimate_length = estimate_length
        self.same_length=same_length

        # kernel part
        self.pos1 = pos1
        self.pos2 = pos2
        self.split = split
        self.N01 = N01


        self.sparse=sparse
        self.cut_off=cut_off
        self.KNN=KNN

        # omics weight part
        if alpha_list is None:
            if not omics_weight:
                self.alpha_list = np.array([1 for _ in range(len(Y_list))])
            else:
                self.alpha_list = np.max(self.m_list) / np.array(self.m_list)
        else:
            self.alpha_list = alpha_list

        self.K = None
        self.K_inv = None
        self.Z = None
        self.U = None
        self.lbds = None
        self.gamma_hat = None
        self.W_hat_list = []
        self.sigma_hat_sqr_list = []

  
        if(KNN):
            points = pos1
            # 查找最近邻
            n_neighbors = KNN_n
            nn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
            nn.fit(points)
            distances, indices = nn.kneighbors(points)

            # 构建原邻接矩阵
            adj_matrix = np.zeros((self.n, self.n), dtype=int)
            for a in range(self.n):
                for b in indices[a]:
                    adj_matrix[b, a] = 1

            # 生成互为邻居的矩阵
            self.A = adj_matrix * adj_matrix.T  # 或用np.logical_and
            lbd, _ = eigh(self.A)
            self.A=(self.A-min(lbd)*np.eye(self.n))/(1-min(lbd))

        
        # logger.info(f"SMOPCA object created, with {self.n} cells and {[Y.shape[0] for Y in self.Y_list]} features")

    def buildKernel(self, length_scale, nu=1.5):
        """
        :param method: implementation of gaussian kernel, recommend sklearn
        :param length_scale: matern kernel length scale, or gaussian/tsne kernel gamma, or cauchy kernel sigma
        :param check_numeric_stability: check if kernel matrix is numerically stable for the following calculations
        """
        
        self.K = []
        self.U = []
        self.lbds = []
        self.length_scale = length_scale

        
        logger.info(f"calculating {self.kernel} kernel, split={self.split}, length_scale = {[np.round(length_scale[i], 3) for i in range(self.d)]}")

        for i in range(self.d):
            if i < self.split:
                pos = self.pos1
            elif i >= self.split:
                
                pos = self.pos2
            if pos is not None:
                
                if self.kernel == "Matern":
                    matern_obj = Matern(length_scale=length_scale[i], nu=nu)
                    K = matern_obj(X=pos, Y=pos)
                elif self.kernel == "rbf":
                    K = rbf_kernel(pos, pos, gamma = 1 / length_scale[i])
                elif self.kernel == "cauchy":
                    squared_diff = distance_matrix(pos, pos) ** 2
                    K = 1 / (1 + squared_diff / length_scale[i] ** 2)
            else:
                
                K = np.identity(self.n)
            if(self.KNN and i >= self.split):
                K=K*self.A
                #lbd, _ = eigh(K)
                #K=(K-min(lbd)*np.eye(self.n))/(1-min(lbd))

            self.K.append(K)

            lbds, U = eigh(K)
            self.U.append(U)
            self.lbds.append(lbds)


        self.K = np.array(self.K)  # (20, 3639, 3639)

        if(self.sparse):
            self.K[np.abs(self.K)<self.cut_off]=0
        self.lbds = np.array(self.lbds)
        self.U = np.array(self.U)
        
        print(self.length_scale)

    def estimateParams(self, W_init=None,iteration_length_scale=2, iterations_sigma_W=20, tol_length_scale=1e-3, 
                       length_scale_Bound=(0.1,20), tol_sigma=1e-2, sigma_init_list=(), sigma_xtol_list=()):
        """
        :param iterations_gamma: number of iterations for gamma
        :param iterations_sigma_W: number of iterations for sigma and W
        :param tol_gamma: tolerance for gamma estimation
        :param tol_sigma: tolerance for sigma estimation
        :param estimate_gamma: choose if kernel length scale needs to be estimated (a bit slower) or fixed as gamma_init
        :param gamma_init: init value for kernel length scale
        :param gamma_bound: bound for estimate gamma
        :param sigma_init_list: init value for sigma, should include the same number of values as the number of modalities
        :param sigma_xtol_list: xtol parameter for brentq function, should include the same number of values as the number of modalities
        :param gamma_tol: tol parameter for minimize_scalar function
        """
        assert len(sigma_init_list) == len(sigma_xtol_list) == self.modality_num
        # logger.info("start estimating parameters, this will take a while...")

        if(W_init):
            # -----------------------------estimate sigma_k------------------------------------
            bound_list = [None for _ in range(self.modality_num)]
            self.sigma_hat_sqr_list=[]
            self.W_hat_list = W_init

            for modality in range(self.modality_num):
                Y = self.Y_list[modality]
                tr_YY_T = np.trace(Y @ Y.T)
                W_hat=self.W_hat_list[modality]
                
                
                def jac_sigma_sqr(_sigma_sqr):  # derivative of -log likelihood w.r.t. sigma_k^2                        
                    W = W_hat.T
                    L = np.array([np.diag((self.lbds * (2 * _sigma_sqr + self.lbds) / (self.lbds + _sigma_sqr) ** 2)[i]) for i in range(self.d)])
                    jac = self.m_list[modality] * self.n / _sigma_sqr - \
                        np.sum(self.lbds / (self.lbds + _sigma_sqr)) / _sigma_sqr ** 2 - tr_YY_T / _sigma_sqr ** 2 + \
                        np.sum(W.reshape(self.d, 1, self.m_list[modality]) @ Y @ self.U @ L @ self.U.swapaxes(1,2) @ Y.T @ W.reshape(self.d, self.m_list[modality], 1) / _sigma_sqr**2)
                    
                    return jac

                def objective(_sigma_sqr):
                    W=W_hat.T
                    L=np.array([np.diag((self.lbds/(self.lbds+_sigma_sqr))[i]) for i in range(self.d)]) 
                    WYU= W.reshape(self.d,1,self.m_list[modality])@Y@self.U
                    part1=np.sum(WYU@L@(WYU).swapaxes(1,2))
                    return self.m_list[modality]*self.n*np.log(_sigma_sqr)+np.sum(np.linalg.slogdet(self.K/_sigma_sqr+np.eye(self.n))[1])+ \
                    (tr_YY_T-part1)/_sigma_sqr

                # estimate a bound for tighter searching range
                if bound_list[modality] is None:
                    lb = ub = 0.1
                    lb_res = -np.inf
                    ub_res = np.inf
                    for sigma in np.arange(0.1, 10.0, 0.1):
                        res = jac_sigma_sqr(sigma)
                        if res < 0:
                            lb = sigma
                            lb_res = res
                        else:
                            ub = sigma
                            ub_res = res
                            break
                    if abs(lb_res) < 1000:  # for a safer bound since this is a bound dependent on last iteration (init values)
                        lb -= 0.05
                    if abs(ub_res) < 1000:
                        ub += 0.05
                    bound_list[modality] = (lb, ub)
                
                try:
                    sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1],
                                        xtol=sigma_xtol_list[modality])
                except:

                    sigma_hat_sqr=scipy.optimize.minimize(objective,sigma_init_list[modality],bounds=[(bound_list[modality][0],bound_list[modality][1])])['x'][0]

                
                sigma_sqr = sigma_hat_sqr
                self.sigma_hat_sqr_list.append(sigma_sqr)

            # -----------------------------estimate length_scale------------------------------------

            def objective(length_scale):  # 第i个length scale的目标函数
                if self.pos is not None:
                    if self.kernel == "rbf":
                        K = rbf_kernel(self.pos, self.pos, gamma=1 / length_scale)
                    elif(self.kernel == "Matern"):
                        matern_obj = Matern(length_scale=length_scale, nu=1.5)
                        K = matern_obj(X=self.pos, Y=self.pos)
                    elif(self.kernel == "cauchy"):  
                        squared_diff = distance_matrix(self.pos, self.pos) ** 2
                        K = 1 / (1 + squared_diff / length_scale ** 2)
                else:
                    K = np.identity(self.n)

                if(self.KNN and l >= self.split):
                    K=K*self.A
                    #lbd, _ = eigh(K)
                    #K=(K-min(lbd)*np.eye(self.n))/(1-min(lbd))

                if(self.sparse):
                    K[np.abs(K)<self.cut_off]=0  

                lbds, U = eigh(K)
                total = 0
                
                for modality in range(self.modality_num):
                    WYU = self.W_hat_list[modality][:, l] @ self.Y_list[modality] @ U 
                    total += self.alpha_list[modality] * (
                        np.sum(np.log(lbds / self.sigma_hat_sqr_list[modality] + 1)) - 
                        WYU @ np.diag(lbds / (lbds + self.sigma_hat_sqr_list[modality])) @ WYU.T / self.sigma_hat_sqr_list[modality]
                    )
                return total

            if self.estimate_length:
                for l in range(self.d):
                    if l < self.split:
                        self.pos = self.pos1
                    else:
                        self.pos = self.pos2
                    
                    if(self.same_length==False):
                        self.length_scale[l] = scipy.optimize.minimize(objective, x0=self.length_scale[l], tol=tol_length_scale, 
                                                                    bounds=Bounds(length_scale_Bound[0], length_scale_Bound[1]))['x']
                    else:
                        if(l==0):
                            self.length_scale[l] = scipy.optimize.minimize(objective, x0=self.length_scale[l], tol=tol_length_scale, 
                                                                    bounds=Bounds(length_scale_Bound[0], length_scale_Bound[1]))['x']                   
                        else:
                            self.length_scale[l]=self.length_scale[0]

                self.buildKernel(length_scale=self.length_scale)


    #----------------------------------------------------------------------------------
    
        for iteration in range(iteration_length_scale):
            bound_list = [None for _ in range(self.modality_num)]
            self.W_hat_list = []
            self.sigma_hat_sqr_list = []
            for modality in range(self.modality_num):  # estimate sigma and W
                Y = self.Y_list[modality]
                tr_YY_T = np.trace(Y @ Y.T)
                sigma_sqr = sigma_init_list[modality]
                sigma_hat_sqr = None
                W_hat = None
                # logger.info(f"estimating sigma{modality + 1}")
                for iter2 in range(iterations_sigma_W):                    
                    # estimate W_k             
                    start = time.time()
                    L = np.array([np.diag((self.lbds / (self.lbds + sigma_sqr))[i]) for i in range(self.d)])
                    A = Y @ self.U @ L @ self.U.swapaxes(1, 2) @ Y.T
                    W_hat = manopt_minimize(-A, self.m_list[modality], self.d)  # 没看懂
                    end = time.time()
                    #print(f"W耗时: {end - start:.4f} 秒")
                    
                    assert W_hat.shape == (self.m_list[modality], self.d)

                    # estimate sigma_k
                    def jac_sigma_sqr(_sigma_sqr):  # derivative of -log likelihood w.r.t. sigma_k^2                        
                        W = W_hat.T
                        L = np.array([np.diag((self.lbds * (2 * _sigma_sqr + self.lbds) / (self.lbds + _sigma_sqr) ** 2)[i]) for i in range(self.d)])
                        jac = self.m_list[modality] * self.n / _sigma_sqr - \
                            np.sum(self.lbds / (self.lbds + _sigma_sqr)) / _sigma_sqr ** 2 - tr_YY_T / _sigma_sqr ** 2 + \
                            np.sum(W.reshape(self.d, 1, self.m_list[modality]) @ Y @ self.U @ L @ self.U.swapaxes(1,2) @ Y.T @ W.reshape(self.d, self.m_list[modality], 1) / _sigma_sqr**2)
                        
                        return jac

                    def objective(_sigma_sqr):
                        W=W_hat.T
                        L=np.array([np.diag((self.lbds/(self.lbds+_sigma_sqr))[i]) for i in range(self.d)]) 
                        WYU= W.reshape(self.d,1,self.m_list[modality])@Y@self.U
                        part1=np.sum(WYU@L@(WYU).swapaxes(1,2))
                        return self.m_list[modality]*self.n*np.log(_sigma_sqr)+np.sum(np.linalg.slogdet(self.K/_sigma_sqr+np.eye(self.n))[1])+ \
                        (tr_YY_T-part1)/_sigma_sqr

                    # estimate a bound for tighter searching range
                    if bound_list[modality] is None:
                        lb = ub = 0.1
                        lb_res = -np.inf
                        ub_res = np.inf
                        for sigma in np.arange(0.1, 10.0, 0.1):
                            res = jac_sigma_sqr(sigma)
                            if res < 0:
                                lb = sigma
                                lb_res = res
                            else:
                                ub = sigma
                                ub_res = res
                                break
                        if abs(lb_res) < 1000:  # for a safer bound since this is a bound dependent on last iteration (init values)
                            lb -= 0.05
                        if abs(ub_res) < 1000:
                            ub += 0.05
                        bound_list[modality] = (lb, ub)
                        logger.info("sigma{} using bound: ({:.5f}, {:.5f})".format(modality + 1, lb, ub))

                    start = time.time()

                    try:
                        sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1],
                                            xtol=sigma_xtol_list[modality])
                    except:

                        sigma_hat_sqr=scipy.optimize.minimize(objective,sigma_init_list[modality],bounds=[(bound_list[modality][0],bound_list[modality][1])])['x'][0]

                    end = time.time()
            
                    #print(f"sigma耗时: {end - start:.4f} 秒")

                    logger.info("iter {} sigma{} brentq done, sigma{}sqr = {:.5f}, sigma{}hatsqr = {:.5f}".format(
                        iter2, modality + 1, modality + 1, sigma_sqr, modality + 1, sigma_hat_sqr))

                    if abs(sigma_sqr - sigma_hat_sqr) < tol_sigma:
                        logger.info(f"reach tolerance threshold, sigma{modality + 1} done!")
                        self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                        self.W_hat_list.append(W_hat)
                        break
                    sigma_sqr = sigma_hat_sqr
                    if iter2 == iterations_sigma_W - 1:
                        logger.warning(f"reach end of iteration for sigma{modality + 1}!")
                        self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                        self.W_hat_list.append(W_hat)

            # estimate length_scale
            start = time.time()
            def objective(length_scale):  # 第i个length scale的目标函数
                if self.pos is not None:
                    
                    if self.kernel == "rbf":
                        K = rbf_kernel(self.pos, self.pos, gamma=1 / length_scale)
                    elif(self.kernel == "Matern"):
                        matern_obj = Matern(length_scale=length_scale, nu=1.5)
                        K = matern_obj(X=self.pos, Y=self.pos)
                    elif(self.kernel == "cauchy"):  
                        squared_diff = distance_matrix(self.pos, self.pos) ** 2
                        K = 1 / (1 + squared_diff / length_scale ** 2)
                else:
                    
                    K = np.identity(self.n)

                if(self.KNN and l >= self.split):
                    K=K*self.A
                    #lbd, _ = eigh(K)
                    #K=(K-min(lbd)*np.eye(self.n))/(1-min(lbd))

                if(self.sparse):
                    K[np.abs(K)<self.cut_off]=0     

                lbds, U = eigh(K)
                total = 0
                
                for modality in range(self.modality_num):
                    WYU = self.W_hat_list[modality][:, l] @ self.Y_list[modality] @ U 
                    total += self.alpha_list[modality] * (
                        np.sum(np.log(lbds / self.sigma_hat_sqr_list[modality] + 1)) - 
                        WYU @ np.diag(lbds / (lbds + self.sigma_hat_sqr_list[modality])) @ WYU.T / self.sigma_hat_sqr_list[modality]
                    )
                return total

            if self.estimate_length:
                for l in range(self.d):
                    
                    if l < self.split:
                       
                        self.pos = self.pos1
                    else:
                      
                        self.pos = self.pos2
                    
                    if(self.same_length==False):
                        
                        self.length_scale[l] = scipy.optimize.minimize(objective, x0=self.length_scale[l], tol=tol_length_scale, 
                                                                    bounds=Bounds(length_scale_Bound[0], length_scale_Bound[1]))['x']
                        
                    else:
                        if(l==0):
                            self.length_scale[l] = scipy.optimize.minimize(objective, x0=self.length_scale[l], tol=tol_length_scale, 
                                                                    bounds=Bounds(length_scale_Bound[0], length_scale_Bound[1]))['x']                   
                        else:
                            self.length_scale[l]=self.length_scale[0]
                    #print("第", l, "个length_scale:", self.length_scale[l])
                end = time.time()
                
                #print(f"length_scale耗时: {end - start:.4f} 秒")
                self.buildKernel(length_scale=self.length_scale)
            else:
                break

        logger.info("estimation complete!")
        for modality, sigma_hat_sqr in enumerate(self.sigma_hat_sqr_list):
            logger.info("sigma{}hatsqr = {:.5f}".format(modality + 1, sigma_hat_sqr))

    def calculatePosterior(self):
        """
        :return: posterior mean of shape (#cells, zdim)
        """
        logger.info("calculating posterior")
        self.K_inv = np.linalg.inv(self.K)
        self.sigma_hat_sqr_list = np.array(self.sigma_hat_sqr_list)
        A = (np.sum(self.alpha_list / self.sigma_hat_sqr_list) * np.eye(self.n) + self.K_inv) / 2
        b = np.zeros((self.n, self.d))
        for modality in range(self.modality_num):
            b += self.alpha_list[modality] / self.sigma_hat_sqr_list[modality] * (self.Y_list[modality].T) @ self.W_hat_list[modality]
        self.Z = 0.5 * np.linalg.inv(A) @ (b.T.reshape(self.d, self.n, 1))
        self.Z = self.Z.squeeze(2)   
        return self.Z.T

    def stepZ(self):
        A = (np.sum(self.alpha_list / self.sigma_hat_sqr_list) * np.eye(self.n) + self.K_inv) / 2
        b = np.zeros((self.n, self.d))
        for modality in range(self.modality_num):
            b += self.alpha_list[modality] / self.sigma_hat_sqr_list[modality] * (self.Y_list[modality].T) @ self.W_hat_list[modality]
        self.Z = 0.5 * np.linalg.inv(A) @ (b.T.reshape(self.d, self.n, 1))
        self.Z = self.Z.squeeze(2)   

        return self.Z.T

    def stepW(self):
        self.W_hat_list = []
 
        for modality in range(self.modality_num):  # estimate sigma and W
            sigma_sqr=self.sigma_hat_sqr_list[modality]
            Y = self.Y_list[modality]
            U,_,Vh=np.linalg.svd(Y@self.Z.T,full_matrices=False)
            self.W_hat_list.append(U@Vh)


class SMOPCA:
    def __init__(self, Y_list, pos, Z_dim=20, omics_weight=False, alpha_list=None, intercept=True, kernel_type='matern', nu=1.5):
        """
        :param Y_list: data matrices from different modalities with shape (#feats, #cells)
        :param pos: spatial coordinates with shape (#cells, 2)
        :param Z_dim: dimension of latent factors
        :param omics_weight: choose if using weighted posterior for different modalities
        :param alpha_list: numpy array, weights of different modalities
        :param intercept: whether to use intercept for data with mean structures
        :param kernel_type: type of kernel, default is matern
        :param nu: matern kernel parameter, common value is 0.5, 1.5 or 2.5
        """
        assert all(Y.shape[1] == Y_list[0].shape[1] for Y in Y_list)
        self.Y_list = Y_list
        self.m_list = [Y.shape[0] for Y in Y_list]
        self.n = Y_list[0].shape[1]
        self.d = Z_dim
        self.modality_num = len(Y_list)

        # kernel part
        self.pos = pos
        self.nu = nu
        self.kernel_type = kernel_type

        # intercept and covariate part, simplified for easier inference
        self.intercept = intercept
        if self.intercept:
            self.q_list = [1 for _ in range(len(Y_list))]
            self.X_list = [np.ones((self.n, 1)) for _ in range(len(Y_list))]
            self.M_list = [np.identity(self.n) - X @ np.linalg.inv((X.T @ X)) @ X.T for X in self.X_list]
        else:
            self.q_list = [0 for _ in range(len(Y_list))]
            self.M_list = [np.identity(self.n) for _ in range(len(Y_list))]

        # omics weight part
        if alpha_list is None:
            if not omics_weight:
                self.alpha_list = np.array([1 for _ in range(len(Y_list))])
            else:
                self.alpha_list = np.max(self.m_list) / np.array(self.m_list)
        else:
            self.alpha_list = alpha_list

        self.K = None
        self.K_inv = None
        self.Z = None
        self.U = None
        self.lbds = None
        self.gamma_hat = None
        self.W_hat_list = []
        self.sigma_hat_sqr_list = []
        self.Z = None
        logger.info(f"SMOPCA object created, with {self.n} cells and {[Y.shape[0] for Y in self.Y_list]} features and {self.kernel_type} kernel")

    def buildKernel(self, method="sklearn", length_scale=1.0, check_numeric_stability=False):
        """
        :param method: implementation of gaussian kernel, recommend sklearn
        :param length_scale: matern kernel length scale, or gaussian/tsne kernel gamma, or cauchy kernel sigma
        :param check_numeric_stability: check if kernel matrix is numerically stable for the following calculations
        """
        if self.kernel_type == "gaussian":
            logger.info(f"calculating {self.kernel_type} kernel with {method} implementation, gamma = {length_scale}")
            if method == "sklearn":
                self.K = rbf_kernel(self.pos, self.pos, gamma=1 / length_scale)
            elif method == "scipy":
                self.K = np.exp(-np.power(distance_matrix(self.pos, self.pos), 2) / length_scale)
        elif self.kernel_type == 'matern':
            logger.info(f"calculating {self.kernel_type} kernel, nu = {self.nu}, length_scale = {length_scale}")
            matern_obj = Matern(length_scale=length_scale, nu=self.nu)
            self.K = matern_obj(X=self.pos, Y=self.pos)
        elif self.kernel_type == 'cauchy':
            logger.info(f"calculating {self.kernel_type} kernel, sigma = {length_scale}")
            squared_diff = distance_matrix(self.pos, self.pos) ** 2
            self.K = 1 / (1 + squared_diff / length_scale ** 2)
        elif self.kernel_type == "tsne":
            self.pos *= length_scale
            self.K = np.power(np.power(distance_matrix(self.pos, self.pos), 2) + 1, -1)
        elif self.kernel_type == "dummy":
            logger.info("using Identity as the kernel matrix")
            self.K = np.identity(self.n)
        else:
            logger.error("other kernel type not implemented yet!")
            raise NotImplemented
        logger.debug("performing eigenvalue decomposition on kernel matrix!")
        self.lbds, self.U = eigh(self.K)

        if check_numeric_stability:
            logger.debug("calculating kernel inverse")
            self.K_inv = np.linalg.inv(self.K)
            K_det, K_num, recon_det = np.linalg.det(self.K), np.sum(self.K - np.identity(self.n)), np.linalg.det(self.K @ self.K_inv)
            if recon_det < -1 or recon_det > 1000:
                logger.warning("kernel matrix status: det={:.4f}, K_num={:.4f}, det(KK^-1)={:.4f}\n"
                               "numerical instability is expected, please try smaller gamma or length_scale".format(
                    K_det, K_num, recon_det))
            else:
                logger.debug("kernel matrix status: det={:.4f}, K_num={:.4f}, det(KK^-1)={:.4f}".format(
                    np.linalg.det(self.K), np.sum(self.K - np.identity(self.n)), np.linalg.det(self.K @ self.K_inv)
                ))

    def estimateParams(self, iterations_gamma=10, iterations_sigma_W=20, tol_gamma=1e-2, tol_sigma=1e-5,
                       estimate_gamma=False, gamma_init=1, gamma_bound=(0.1, 5),
                       sigma_init_list=(), sigma_xtol_list=(), gamma_tol=0.1):
        """
        :param iterations_gamma: number of iterations for gamma
        :param iterations_sigma_W: number of iterations for sigma and W
        :param tol_gamma: tolerance for gamma estimation
        :param tol_sigma: tolerance for sigma estimation
        :param estimate_gamma: choose if kernel length scale needs to be estimated (a bit slower) or fixed as gamma_init
        :param gamma_init: init value for kernel length scale
        :param gamma_bound: bound for estimate gamma
        :param sigma_init_list: init value for sigma, should include the same number of values as the number of modalities
        :param sigma_xtol_list: xtol parameter for brentq function, should include the same number of values as the number of modalities
        :param gamma_tol: tol parameter for minimize_scalar function
        """
        assert len(sigma_init_list) == len(sigma_xtol_list) == self.modality_num
        logger.info("start estimating parameters, this will take a while...")

        gamma = gamma_init
        self.buildKernel(length_scale=gamma)

        for iter1 in range(iterations_gamma):
            bound_list = [None for _ in range(self.modality_num)]
            self.W_hat_list = []
            self.sigma_hat_sqr_list = []
            for modality in range(self.modality_num):
                Y = self.Y_list[modality]
                tr_YY_T = np.trace(Y @ Y.T)
                sigma_sqr = sigma_init_list[modality]
                sigma_hat_sqr = None
                W_hat = None
                logger.info(f"estimating sigma{modality + 1}")
                for iter2 in range(iterations_sigma_W):
                    # estimate W_k
                    D1 = np.diag(self.lbds * sigma_sqr / (self.lbds + sigma_sqr))
                    P1 = Y @ self.U
                    G = P1 @ D1 @ P1.T
                    vals, vec = eigh(G)
                    W_hat = vec[:, -self.d:]  # eigenvectors w.r.t. d largest eigenvalues
                    assert W_hat.shape == (self.m_list[modality], self.d)

                    # estimate sigma_k
                    def jac_sigma_sqr(_sigma_sqr):  # derivative of -log likelihood w.r.t. sigma_k^2
                        part1 = self.m_list[modality] * self.n / _sigma_sqr
                        part2 = -np.sum(self.lbds / (self.lbds + _sigma_sqr)) * self.d / _sigma_sqr
                        D2 = np.diag((self.lbds * (2 * _sigma_sqr + self.lbds)) / (self.lbds + _sigma_sqr) ** 2)
                        P2 = W_hat.T @ Y @ self.U
                        part3 = (np.trace(P2 @ D2 @ P2.T) - tr_YY_T) / _sigma_sqr ** 2
                        jac = part1 + part2 + part3
                        #logger.debug("jac{}({:.5f}) = {:.5f}".format(modality + 1, _sigma_sqr, jac))
                        return jac

                    # estimate a bound for tighter searching range
                    if bound_list[modality] is None:
                        lb = ub = 0.1
                        lb_res = -np.inf
                        ub_res = np.inf
                        for sigma in np.arange(0.1, 10.0, 0.1):
                            res = jac_sigma_sqr(sigma)
                            if res < 0:
                                lb = sigma
                                lb_res = res
                            else:
                                ub = sigma
                                ub_res = res
                                break
                        if abs(lb_res) < 1000:  # for a safer bound since this is a bound dependent on last iteration (init values)
                            lb -= 0.05
                        if abs(ub_res) < 1000:
                            ub += 0.05
                        bound_list[modality] = (lb, ub)
                        logger.info("sigma{} using bound: ({:.5f}, {:.5f})".format(modality + 1, lb, ub))
                    try:
                    #sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1],
                                           #xtol=sigma_xtol_list[modality])
                        sigma_hat_sqr = brentq(jac_sigma_sqr, bound_list[modality][0], bound_list[modality][1],
                                           xtol=sigma_xtol_list[modality])
                    except:

                        sigma_hat_sqr=scipy.optimize.minimize(jac_sigma_sqr,sigma_sqr)['x'][0]


                    #logger.info("iter {} sigma{} brentq done, sigma{}sqr = {:.5f}, sigma{}hatsqr = {:.5f}".format(
                        #iter2, modality + 1, modality + 1, sigma_sqr, modality + 1, sigma_hat_sqr))

                    if abs(sigma_sqr - sigma_hat_sqr) < tol_sigma:
                        logger.info(f"reach tolerance threshold, sigma{modality + 1} done!")
                        self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                        self.W_hat_list.append(W_hat)
                        break
                    sigma_sqr = sigma_hat_sqr
                    if iter2 == iterations_sigma_W - 1:
                        logger.warning(f"reach end of iteration for sigma{modality + 1}!")
                        self.sigma_hat_sqr_list.append(sigma_hat_sqr)
                        self.W_hat_list.append(W_hat)

            if not estimate_gamma:
                break

            def f_gamma(g):
                matern_obj = Matern(length_scale=g, nu=self.nu)
                K = matern_obj(X=self.pos, Y=self.pos)
                lbds, U = eigh(K)
                val = 0
                for k in range(self.modality_num):
                    if k == 0:
                        continue
                    alpha_k = self.alpha_list[k]
                    sigma_k_sqr = self.sigma_hat_sqr_list[k]
                    W_k = self.W_hat_list[k]
                    Y_k = self.Y_list[k]
                    part1 = self.d * np.sum(np.log(1 + lbds / sigma_k_sqr))
                    D = np.diag(lbds / (lbds + sigma_k_sqr))
                    part2 = -np.trace(W_k.T @ Y_k @ U @ D @ U.T @ Y_k.T @ W_k) / sigma_k_sqr
                    val += alpha_k * (part1 + part2)
                logger.debug("f_gamma({:.5f}) = {:.5f}".format(g, val))
                return val

            ret = scipy.optimize.minimize_scalar(f_gamma, method="Bounded", bounds=gamma_bound, tol=gamma_tol)
            gamma_hat = ret['x']
            logger.info("iter {} gamma minimize done, gamma = {:.5f}, gamma_hat = {:.5f}".format(iter1, gamma, gamma_hat))
            self.buildKernel(length_scale=gamma_hat)
            if abs(gamma - gamma_hat) < tol_gamma:
                self.gamma_hat = gamma_hat
                logger.info(f"reach tolerance threshold, gamma done!")
                break
            gamma = gamma_hat
            if iter1 == iterations_gamma - 1:
                self.gamma_hat = gamma_hat
                logger.warning(f"reach end of iteration for gamma!")
                break

        logger.info("estimation complete!")
        for modality, sigma_hat_sqr in enumerate(self.sigma_hat_sqr_list):
            logger.info("sigma{}hatsqr = {:.5f}".format(modality + 1, sigma_hat_sqr))
        if estimate_gamma:
            logger.info("gamma_hat = {:.5f}".format(self.gamma_hat))

    def calculatePosterior(self):
        """
        :return: posterior mean of shape (#cells, zdim)
        """
        logger.info("calculating posterior")
        self.sigma_hat_sqr_list = np.array(self.sigma_hat_sqr_list)
        c = np.sum(self.alpha_list / self.sigma_hat_sqr_list)
        D = np.diag(self.lbds / (1 + self.lbds * c))
        A_inv = self.U @ D @ self.U.T
        B = 0
        for modality in range(self.modality_num):
            B += ((self.alpha_list[modality] / self.sigma_hat_sqr_list[modality]) * self.M_list[modality] @ self.Y_list[modality].T @ self.W_hat_list[modality])
        self.Z = (A_inv @ B).T
        return self.Z.T


