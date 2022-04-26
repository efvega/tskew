# tskew

This package contains an implementation of the the skew-t distribution of Azzalini [(link)](https://doi.org/10.1111/1467-9868.00391). Its PDF is given by 
<img src="https://render.githubusercontent.com/render/math?math={    f_X(x) = 2 t ( x, \nu) T_1 \left(\alpha^\intercal \omega^{-1} (x - \mu) \left( \frac{\nu %2B  d}{Q_x %2B  \nu} \right), \nu %2B d \right) }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white}    f_X(x) = 2 t ( x, \nu) T_1 \left(\alpha^\intercal \omega^{-1} (x - \mu) \left( \frac{\nu %2B  d}{Q_x %2B  \nu} \right), \nu %2B d \right) }#gh-dark-mode-only"> 

where 

<img src="https://render.githubusercontent.com/render/math?math={   \omega = \diag (\Omega) }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white}    \omega = \diag (\Omega)  }#gh-dark-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={   Q_x = (x - \mu)^\intercal \Omega^{-1}(x - \mu) }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white}    Q_x = (x - \mu)^\intercal \Omega^{-1}(x - \mu)  }#gh-dark-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={   t_d(x, \nu) = \frac{\Gamma(\frac{\nu %2B d}{2})}{\Gamma \left( \frac{\nu}{2} \right) (\pi \nu)^{d/2} \left|\Omega \right|^{1/2}} \left(1 %2B \frac{Q_y}{\nu} \right)^{-\frac{\nu %2B d}{2}} }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white}    t_d(x, \nu) = \frac{\Gamma(\frac{\nu %2B d}{2})}{\Gamma \left( \frac{\nu}{2} \right) (\pi \nu)^{d/2} \left|\Omega \right|^{1/2}} \left(1 %2B \frac{Q_y}{\nu} \right)^{-\frac{\nu %2B d}{2}}  }#gh-dark-mode-only">

<img src="https://render.githubusercontent.com/render/math?math={   T_1(x, r) = \frac{1}{2} %2B \frac{x \Gamma \left( \frac{r %2B 1}{2} \right)}{\Gamma \left( \frac{r}{2} \right) (\pi r)^{d/2}} \cdot {}_{2}F_{1} \left( \frac{1}{2}, \frac{r %2B 1}{2}, \frac{3}{2}, -\frac{x^2}{r}  \right) }#gh-light-mode-only">
<img src="https://render.githubusercontent.com/render/math?math={\color{white}   T_1(x, r) = \frac{1}{2} %2B \frac{x \Gamma \left( \frac{r %2B 1}{2} \right)}{\Gamma \left( \frac{r}{2} \right) (\pi r)^{d/2}} \cdot {}_{2}F_{1} \left( \frac{1}{2}, \frac{r %2B 1}{2}, \frac{3}{2}, -\frac{x^2}{r}  \right)  }#gh-dark-mode-only">.



Here we limit our attention to the 1-D version. A simplified description of the parameters goes like

<img src="https://render.githubusercontent.com/render/math?math={    \mu }#gh-light-mode-only"> <img src="https://render.githubusercontent.com/render/math?math={\color{white}    \mu }#gh-dark-mode-only"> - Location parameter (not necessarily the mean). 

<img src="https://render.githubusercontent.com/render/math?math={    \Omega }#gh-light-mode-only"> <img src="https://render.githubusercontent.com/render/math?math={\color{white}    \Omega }#gh-dark-mode-only"> - Scale parameter (not necessarily the variance or standard deviation). 

<img src="https://render.githubusercontent.com/render/math?math={    \alpha }#gh-light-mode-only"> <img src="https://render.githubusercontent.com/render/math?math={\color{white}    \alpha }#gh-dark-mode-only"> - Skewness parameter (not necessarily the skewness itself). 

<img src="https://render.githubusercontent.com/render/math?math={    \nu }#gh-light-mode-only"> <img src="https://render.githubusercontent.com/render/math?math={\color{white}    \nu }#gh-dark-mode-only"> - Tailedness parameter (influences kurtosis). 

## Exploring the code
After cloning the repository, use the following commands to install the necessary requirements:
```
pip install -r path/to/cloned/repo/requirements.txt
```

Then, run the demo script via
```
python -i tskew_demo.py
```
The generated images will be similar to


![PDF_and_CDF](https://github.com/efvega/tskew/blob/main/media/pdf_cdf.png?) ![Exponential_fit](https://github.com/efvega/tskew/blob/main/media/exponential_fit.png?)



The -i flag makes the script run in interactive mode so that the plots do not immediately close on program termination. A simplified use-case might look like
```
from tskew.tskew import fit_tskew
loc_est, scale_est, df_est, skew_param_est = fit_tskew(my_data)
```

