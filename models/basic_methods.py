import numpy as np
from typing import List
import scipy.stats as ss



class DeltaAdditive:
    def __init__(self) -> None:
        """Delta insights is suitable for additive data
        """
        pass

    @staticmethod
    def delta_analysis_helper(x_curr, x_base, y_curr, y_base):
        assert y_curr != y_base
        return np.round((x_curr - x_base)/(y_curr - y_base), 3)
    
    @staticmethod
    def analysis( 
                    x_curr: np.ndarray, 
                    x_base: np.ndarray,
                    y_curr: np.float32,
                    y_base: np.float32):
        assert len(x_curr) == len(x_base)
        assert y_curr != y_base
        res = DeltaAdditive.delta_analysis_helper(x_curr, x_base, y_curr, y_base)
        scaler = np.sum(res)
        assert scaler != 0
        return np.round(res/scaler, 3)
    

class MultiplicativeLogarrithm:
    """All the input value should be POSITIVE!
    If it is 0, then add a fix value to make all of them POSITIVE!
    """
    def __init__(self) -> None:
        pass

    @staticmethod
    def analysis(x_curr:np.ndarray, 
                    x_base:np.ndarray, 
                    y_curr:np.ndarray, 
                    y_base:np.ndarray):
        assert len(x_curr) == len(x_base)
        res = dict()
        # we need all values of x_curr and x_base > 0
        # also need the y_curr and y_base >0
        assert np.all(x_curr >0)
        assert np.all(x_base >0)
        assert y_curr > 0
        assert y_base > 0
        
        x_curr = np.log(np.clip(x_curr, a_min=1e-5, a_max=None))
        x_base = np.log(np.clip(x_curr, a_min=1e-5, a_max=None))
        y_curr = np.log(y_curr)
        y_base = np.log(y_base)

        delta_method=DeltaAdditive()
        res = delta_method.analysis(x_curr, x_base, y_curr, y_base)
        return res
    

class Division:
    def __init__(self):
        pass
    @staticmethod
    def control_variant(x_curr_nom, x_curr_denom,
                        x_base_nom, x_base_denom, 
                        y_curr, y_base):
        def analysis_helper(y_sub, y_curr, y_base):
            den = y_curr - y_base
            nu = y_curr - y_sub
            return nu/den
        res = []
        for i in range(len(x_curr_denom)):
            curr_deno_i = x_curr_denom[i]
            curr_nomi_i = x_curr_nom[i]
            base_deno_i = x_base_denom[i]
            base_nomi_i = x_base_nom[i]
            
            x_curr_nom[i] = base_nomi_i
            x_curr_denom[i] = base_deno_i

            y_sub = np.sum(x_curr_nom)/np.sum(x_curr_denom)
            res.append(analysis_helper(y_sub, y_curr, y_base))

            x_curr_nom[i] = curr_nomi_i
            x_curr_denom[i] = curr_deno_i

        scaler = np.sum(np.abs(res))
        res = np.asarray(res)/scaler
        return res
    
    # https://www.volcengine.com/docs/4726/1217644 组合指标的算法说明
    @staticmethod
    def decomposition(x_curr_nom, x_curr_denom,
                    x_base_nom, x_base_denom, y_curr, y_base):
        x_curr_ratio = x_curr_nom/x_curr_denom
        x_base_ratio = x_base_nom/x_base_denom
        diff_nom_ratio = (x_curr_ratio - x_base_ratio) * x_curr_denom / np.sum(x_base_denom)
        diff_denom_ratio = (x_base_ratio - y_base) * (x_curr_denom - x_base_denom)/np.sum(x_curr_denom)
        score = diff_nom_ratio + diff_denom_ratio
        res = np.round(score/(y_curr - y_base),3)
        res /= np.sum(np.abs(res))
        return res


    @staticmethod
    def analysis(x_curr_nominator:np.ndarray,
                    x_curr_denom:np.ndarray, 
                    x_base_nominator:np.ndarray,
                    x_base_denom:np.ndarray,
                    y_curr:np.float32,
                    y_base:np.float32,
                    method='control_variant'
                 ):
        assert method in ["control_variant", "decomposition", "compound"]

        if method in ['control_variant', 'compound']:
            res = Division.control_variant(x_curr_nominator, x_curr_denom,
                        x_base_nominator, x_base_denom, 
                        y_curr, y_base)
            
        elif method == 'decomposition':
            res = Division.decomposition(x_curr_nominator, x_curr_denom,
                        x_base_nominator, x_base_denom, 
                        y_curr, y_base)
        else:
            pass
        
        return np.round(res,3)

# unit test
if __name__ == "__main__":
    x_data = np.random.random((50,10))
    x_curr = x_data[5,:] * 10
    x_base = x_data[1,:] * 10

    # Test DeltaAdditive
    delta_test = DeltaAdditive()
    y_curr = np.sum(x_curr)
    y_base = np.sum(x_base)
    delta_res = delta_test.analysis(x_curr, x_base, y_curr, y_base)
    
    # Product test
    product_test = MultiplicativeLogarrithm()
    y_curr = np.product(x_curr)
    y_base = np.product(x_base)
    prod_res = product_test.analysis(x_curr, x_base, y_curr, y_base)

    # Division Test
    x_curr_denom = x_data[5,:] * 20 + 4
    x_curr_nom = x_data[10,:]*5 + 6
    x_base_denom = x_data[6,:]*20 + 1
    x_base_nom = x_data[16,:]*5 + 3
    y_curr = np.sum(x_curr_nom)/np.sum(x_curr_denom)
    y_base = np.sum(x_base_nom)/np.sum(x_base_denom)


    # Div 1
    div_analysis = Division()
    control_res = div_analysis.analysis(x_curr_nom, x_curr_denom, x_base_nom, x_base_denom, y_curr, y_base, method='control_variant')
    decomp_res = div_analysis.analysis(x_curr_nom, x_curr_denom, x_base_nom, x_base_denom, y_curr, y_base, method='decomposition')

    print(f"Delta Result: {delta_res}\nProduct Res: {prod_res}\nDiv-1 Res:{control_res}\nDiv-2 Res: {decomp_res}")