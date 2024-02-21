import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn import preprocessing
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from scipy.sparse import linalg
from scipy.special import softmax as scipy_softmax

lbin = preprocessing.LabelBinarizer()

def ppi_multi_class_pointestimate(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    lhat = None,
    coord = None,
    optimizer_options=None,
):
    """Computes the prediction-powered point estimate of the multiclass logistic regression coefficients.

    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        lhat (float, optional): Power-tuning parameter. The default value `None` will estimate the optimal value from data. Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical point estimate.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, optional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
       
    Returns:
        ndarray: Prediction-powered point estimate of the multiclass logistic regression coefficients.

    """
    
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    
    classes = np.sort(np.unique(Y))[::-1]
    K = len(classes)
    
    if "ftol" not in optimizer_options.keys():
        optimizer_options = {"ftol": 1e-15}
        
    
    # Helper function to obtain EY
    def get_EY(K, d):
        out = np.zeros((K * d, K * d))
        for k in range(K):

            Ey = np.zeros((d, K * d))
            Ey[:, k * d:(k + 1) * d] = np.eye(d)
            out[k * d:(k + 1) * d, :] = Ey
        out = np.reshape(out,(K, d, K*d))
        return out

    
    # Rectified multiclass logistic loss function
    def rectified_multiclass_logistic_loss(_theta):
        
        # by default class 0 is the reference class
        _theta[0:d] = 0

        EY = get_EY(K,d)
        loss0 = 0
        loss1 = 0
        for i in range(n):
            y = Y[i]
            Xi = X[i,:]
            Ey = EY[y,:,:]
            loss0 +=  -(Xi @ Ey) @ _theta + np.log(np.sum(np.exp(Xi @ EY @ _theta)))

            yhat = Yhat[i]
            Eyhat = EY[yhat,:,:]
            loss1 += -(Xi @ Eyhat) @ _theta + np.log(np.sum(np.exp(Xi @ EY @ _theta)))

        loss2 = 0
        for i in range(N):
            y_unlabeled = Yhat_unlabeled[i]
            Ey_unlabeled = EY[y_unlabeled,:,:]
            Xi_unlabeled = X_unlabeled[i,:]
            loss2 += -(Xi_unlabeled @ Ey_unlabeled) @ _theta + np.log(np.sum(np.exp(Xi_unlabeled @ EY @ _theta)))


        loss = 1 / n * loss0 - lhat_curr / n * loss1 + lhat_curr / N * loss2
        return loss
    
    
    # Helper function to transform theta vector to 2D array [K, d]
    def theta_2d(_theta, K):
        return np.reshape(_theta, (K, -1))

    
    # Helper function to compute the probability matrix using softmax function
    def mysoftmax(_theta_2D, X):
        """
        theta_2D is of shape [K, d] 
        """
        # by default class 0 is the reference class 
        _theta_2D[0,:] = np.zeros(X.shape[1])
        
        a = np.exp(X @ _theta_2D.T)
        s = a / np.sum(a, axis=1, keepdims=True)
        return s
    
    
    # Helper function to compute gradient of the parameter 
    def gradient(X, Y, _theta, K):
        """
        theta is 1D array
        """
        Y_onehot = lbin.fit_transform(Y.reshape(-1,1))
        _theta_2D = theta_2d(_theta, K) 
        
        # by default class 0 is the reference class 
        _theta_2D[0,:] = np.zeros(X.shape[1])
        P = mysoftmax(_theta_2D, X)
        gd = -(Y_onehot - P).T.dot(X)
        
        return gd.ravel()

    # gradient of the rectified loss
    def rectified_multiclass_logistic_grad(_theta):

        return (
            lhat_curr/ N * gradient(X_unlabeled, Yhat_unlabeled, _theta, K)
            - lhat_curr / n * gradient(X, Yhat, _theta, K) +
            1 / n * gradient(X, Y, _theta, K)
        )
   
    # Initialize theta
    # theta_0 = np.random.random((K, d)).ravel()
    theta_0 = (
        LogisticRegression(
            multi_class='multinomial',
            penalty=None,
            solver="lbfgs",
            max_iter=10000,
            tol=1e-15,
            fit_intercept=False,
        )
        .fit(X, Y)
        .coef_.squeeze()
    ).ravel() 
    
    
    lhat_curr = 1 if lhat is None else lhat
    
    # optimization over (K-1)*d degrees of freedom by forcing reference class parameters to zero
    ppi_pointest_extra = minimize(
        fun=rectified_multiclass_logistic_loss, 
        x0=theta_0,
        jac=rectified_multiclass_logistic_grad,
        method="L-BFGS-B",
        tol=optimizer_options["ftol"],
        options=optimizer_options).x
    
    # remove the reference class parameters
    ppi_pointest = np.delete(theta_2d(ppi_pointest_extra, K), 0, 0).flatten(order = "C")
    
    if lhat is None:
        (
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
        ) = _multiclass_ci_get_stats(
            ppi_pointest,
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
        )
        
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        
        return ppi_multi_class_pointestimate(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            optimizer_options=optimizer_options,
            lhat=lhat,
            coord=coord,
        )
      
    else:
        return ppi_pointest
    
    
def _multiclass_ci_get_stats(
    pointest,
    X, 
    Y, 
    Yhat, 
    X_unlabeled, 
    Yhat_unlabeled, 
    use_unlabeled = True,
):
    """Computes the statistics needed for the multiclass logistic regression confidence interval.

    Args:
        pointest (ndarray): Point estimate of the multiclass logistic regression coefficients of (K-1)*d degrees of freedom
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        use_unlabeled (bool, optional): Whether to use the unlabeled data.

    Returns:
        grads (ndarray): Gradient of the loss function on the labeled data.
        grads_hat (ndarray): Gradient of the loss function on the labeled predictions.
        grads_hat_unlabeled (ndarray): Gradient of the loss function on the unlabeled predictions.
        inv_hessian (ndarray): Inverse Hessian of the loss function on the unlabeled data.
    """
    
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    
    classes = np.sort(np.unique(Y))[::-1]
    K = len(classes)
    
    # Helper softmax function to accomodate the (K-1)*d degrees of freedom
    """ modified from Stephen's R code """
    def softmax(_theta, X, K):
        n, p = X.shape
        numer = np.zeros((n, K - 1))

        for class_ in range(K - 1):
            idx = np.arange(class_ * p, (class_ + 1) * p) 
            numer[:, class_] = np.exp(X @ _theta[idx]).reshape(-1)

        denom = numer.sum(axis=1) + 1
        probs = np.hstack((numer, np.ones((n, 1)))) / denom[:, np.newaxis]

        return probs

    
    # Helper function for block diagonal matrix
    def block_diag(*arrs):
        return sparse.block_diag(arrs).toarray()

    
    # 'Meat' for Hessian Matrix
    """ modified from Stephen's R code """
    def block_W(probs):
        K = probs.shape[1]
        n = probs.shape[0]
        W = []
        for class_ in range(K - 1):
            W.append(np.diag(probs[:, class_] * (1 - probs[:, class_])))
        W = block_diag(*W)
        if K > 2:
            for i in range(K - 2):
                for j in range(i + 1, K - 1):
                    row_idx = n * i
                    col_idx = n * j
                    W[row_idx:row_idx+n, col_idx:col_idx+n] = np.diag(-probs[:, i] * probs[:, j])
            W += W.T
            np.fill_diagonal(W, np.diagonal(W) / 2)
        return W

   
    # Function to compute Gradients and Hessian 
    """ modified from Stephen's R code 
        gradients and hessian are not averaged over sample size
    """
    def multiclass_logistic_get_stats(Y, X, _theta):
        
        classes = np.sort(np.unique(Y))[::-1] 
        K = len(classes)
        
        probs = softmax(_theta, X, K)
        probs_vec = probs[:,0:(K-1)].flatten(order='F')
        
        y = np.stack([(Y == class_).astype(float) for class_ in classes[:(K-1)]]).flatten(order = "C")
        X_bmat = sparse.bmat([[X if j == i else None for j in range(K - 1)] for i in range(K - 1)]).toarray()
        W_bmat = block_W(probs)

        grads = X_bmat.T.dot(y - probs_vec)
        hessian = -X_bmat.T.dot(W_bmat).dot(X_bmat)
        return {"grads": grads, "hessian": hessian}
    
    # gradient evaluated for each of the n labeled observations 
    grads = np.zeros((n, (K-1)*d))
    grads_hat = np.zeros((n, (K-1)*d))
    for i in range(n):
        Xi = X[i].reshape(1, d)
        Yi = Y[i]
        Yi_hat = Yhat[i]
        probs = softmax(pointest, Xi, K)
        probs_vec = probs[:,0:(K-1)].flatten(order='F')
        y = np.stack([(Yi == class_).astype(float) for class_ in classes[:(K-1)]]).flatten(order = "C")
        yhat = np.stack([(Yi_hat == class_).astype(float) for class_ in classes[:(K-1)]]).flatten(order = "C")
        X_bmat = sparse.bmat([[Xi if j == i else None for j in range(K - 1)] for i in range(K - 1)]).toarray()
        W_bmat = block_W(probs)
        grads_i = X_bmat.T.dot(y - probs_vec)
        grads_i_hat = X_bmat.T.dot(yhat - probs_vec)

        grads[i, :] = grads_i
        grads_hat[i,:] = grads_i_hat
    

    # gradient evaluated for each of the N unlabeled observations
    grads_hat_unlabeled = np.zeros((N, (K-1)*d))
    if use_unlabeled:
        for i in range(N):
            Xi_unlabeled = X_unlabeled[i].reshape(1, d)
            Yi_hat_unlabeled = Yhat_unlabeled[i]
            probs = softmax(pointest, Xi_unlabeled, K)
            probs_vec = probs[:,0:(K-1)].flatten(order='F')
            y_unlabeled = np.stack([(Yi_hat_unlabeled == class_).astype(float) for class_ in classes[:(K-1)]]).flatten(order = "C")
            X_bmat = sparse.bmat([[Xi_unlabeled if j == i else None for j in range(K - 1)] for i in range(K - 1)]).toarray()
            W_bmat = block_W(probs)
            grads_i = X_bmat.T.dot(y_unlabeled - probs_vec)

            grads_hat_unlabeled[i, :] = grads_i
    
    # stats of labeled data
    stats_X_Y = multiclass_logistic_get_stats(Y, X, pointest)
    
    if use_unlabeled:
        stats_unlabeled = multiclass_logistic_get_stats(Yhat_unlabeled, X_unlabeled, pointest)
        hessian = 1/ (n+N) * (stats_X_Y['hessian'] + stats_unlabeled['hessian'])
    else:
        hessian = stats_X_Y['hessian']
        
    inv_hessian = np.linalg.inv(hessian)

    return grads, grads_hat, grads_hat_unlabeled, hessian, inv_hessian



def ppi_multiclass_logistic_ci(
    X,
    Y,
    Yhat,
    X_unlabeled,
    Yhat_unlabeled,
    alpha=0.1,
    alternative="two-sided",
    lhat=None,
    coord=None,
    optimizer_options=None,
):
    
    """Computes the prediction-powered confidence interval for the multiclass logistic regression coefficients using the PPI++ algorithm 
    Args:
        X (ndarray): Covariates corresponding to the gold-standard labels.
        Y (ndarray): Gold-standard labels.
        Yhat (ndarray): Predictions corresponding to the gold-standard labels.
        X_unlabeled (ndarray): Covariates corresponding to the unlabeled data.
        Yhat_unlabeled (ndarray): Predictions corresponding to the unlabeled data.
        alpha (float, optional): Error level; the confidence interval will target a coverage of 1 - alpha. Must be in the range (0, 1).
        alternative (str, optional): Alternative hypothesis, either 'two-sided', 'larger' or 'smaller'.
        lhat (float, optional): Power-tuning parameter. 
            - The default value `None` will estimate the optimal value from data. 
            - Setting `lhat=1` recovers PPI with no power tuning, and setting `lhat=0` recovers the classical CLT interval.
            - coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        optimizer_options (dict, ooptional): Options to pass to the optimizer. See scipy.optimize.minimize for details.
        
    Returns:
        tuple: Lower and upper bounds of the prediction-powered confidence interval for the multiclass logistic regression coefficients.

    """
    
    n = Y.shape[0]
    d = X.shape[1]
    N = Yhat_unlabeled.shape[0]
    
    classes = np.sort(np.unique(Y))[::-1]
    K = len(classes)
    df = (K-1)*d 
    
    use_unlabeled = lhat != 0
    

    ppi_pointest = ppi_multi_class_pointestimate(
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        lhat=lhat,
        coord=coord,
        optimizer_options=optimizer_options,

    )
    
    grads, grads_hat, grads_hat_unlabeled, hessian, inv_hessian = _multiclass_ci_get_stats(
        ppi_pointest,
        X,
        Y,
        Yhat,
        X_unlabeled,
        Yhat_unlabeled,
        use_unlabeled=use_unlabeled,
    )
    
    if lhat is None:
        lhat = _calc_lhat_glm(
            grads,
            grads_hat,
            grads_hat_unlabeled,
            inv_hessian,
            clip=True,
        )
        return ppi_multiclass_logistic_ci(
            X,
            Y,
            Yhat,
            X_unlabeled,
            Yhat_unlabeled,
            alpha=alpha,
            optimizer_options=optimizer_options,
            alternative=alternative,
            lhat=lhat,
            coord=coord,
        )
    
    
    var_unlabeled = np.cov(lhat * grads_hat_unlabeled.T).reshape(df, df)

    var = np.cov(grads.T - lhat * grads_hat.T).reshape(df, df)

    Sigma_hat = inv_hessian @ (n / N * var_unlabeled + var) @ inv_hessian
    
    ci_res = _zconfint_generic(
        ppi_pointest,
        np.sqrt(np.diag(Sigma_hat) / n),
        alpha=alpha,
        alternative=alternative,)
    
    return {"pointest": ppi_pointest,
            "ci": ci_res, 
            "se":  np.sqrt(np.diag(Sigma_hat) / n), 
            "lhat": lhat}

def _calc_lhat_glm(
    grads, grads_hat, grads_hat_unlabeled, inv_hessian, coord=None, clip=False
):
    """
    Calculates the optimal value of lhat for the prediction-powered confidence interval for GLMs.

    Args:
        grads (ndarray): Gradient of the loss function with respect to the parameter evaluated at the labeled data.
        grads_hat (ndarray): Gradient of the loss function with respect to the model parameter evaluated using predictions on the labeled data.
        grads_hat_unlabeled (ndarray): Gradient of the loss function with respect to the parameter evaluated using predictions on the unlabeled data.
        inv_hessian (ndarray): Inverse of the Hessian of the loss function with respect to the parameter.
        coord (int, optional): Coordinate for which to optimize `lhat`. If `None`, it optimizes the total variance over all coordinates. Must be in {1, ..., d} where d is the shape of the estimand.
        clip (bool, optional): Whether to clip the value of lhat to be non-negative. Defaults to `False`.

    Returns:
        float: Optimal value of `lhat`. Lies in [0,1].
    """
    n = grads.shape[0]
    N = grads_hat_unlabeled.shape[0]
    d = inv_hessian.shape[0]
    cov_grads = np.zeros((d, d))

    for i in range(n):
        cov_grads += (1 / n) * (
            np.outer(
                grads[i] - grads.mean(axis=0),
                grads_hat[i] - grads_hat.mean(axis=0),
            )
            + np.outer(
                grads_hat[i] - grads_hat.mean(axis=0),
                grads[i] - grads.mean(axis=0),
            )
        )
    var_grads_hat = np.cov(
        np.concatenate([grads_hat, grads_hat_unlabeled], axis=0).T
    )

    if coord is None:
        vhat = inv_hessian
    else:
        vhat = inv_hessian @ np.eye(d)[coord]

    if d > 1:
        num = (
            np.trace(vhat @ cov_grads @ vhat)
            if coord is None
            else vhat @ cov_grads @ vhat
        )
        denom = (
            2 * (1 + (n / N)) * np.trace(vhat @ var_grads_hat @ vhat)
            if coord is None
            else 2 * (1 + (n / N)) * vhat @ var_grads_hat @ vhat
        )
    else:
        num = vhat * cov_grads * vhat
        denom = 2 * (1 + (n / N)) * vhat * var_grads_hat * vhat

    lhat = num / denom
    if clip:
        lhat = np.clip(lhat, 0, 1)
    return lhat.item()