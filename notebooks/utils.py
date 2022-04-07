import numpy as np
import pandas as pd
import metnum
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.cm as cm
from IPython.display import HTML
import random
import pandas as pd
import math

def RMSE(aprox, real):
    return np.sqrt(((aprox - real)**2).mean())

def RMSLE(aprox, real):
    return RMSE(np.log(np.abs(aprox) + 1), np.log(np.abs(real) + 1))

def kfold(A, b, k, funcValidation):
    blockSize = int(A.shape[0] / k)
    error = 0
    iteracionesQueDieronNan = 0
    for i in range(k):

        limite_1 = blockSize*i
        limite_2 = min(A.shape[0], blockSize*(i+1))
        """ print("Rangos train: 0-%d y %d-%d"% (limite_1, limite_2, A.shape[0]))
        print("Rangos validation: %d-%d"% (limite_1, limite_2)) """

        A_train, b_train = np.concatenate((A[0:limite_1], A[limite_2:])), np.concatenate((b[0:limite_1], b[limite_2:]))
        A_val, b_val = A[limite_1:limite_2], b[limite_1:limite_2]

        lr = metnum.LinearRegression()
        lr.fit(A_train, b_train)
        b_predict = lr.predict(A_val).flatten()

        """ print("Real")
        print(b_val)
        print("Prediccion")
        print(b_predict) """
        validationResult = funcValidation(b_predict, b_val)
        error += validationResult if not math.isnan(validationResult) else 0
        iteracionesQueDieronNan += 0 if not math.isnan(validationResult) else 1
    if(iteracionesQueDieronNan > 0):
        print("Hubo ", iteracionesQueDieronNan, " iteraciones que dieron NaN")
    return error / (k-iteracionesQueDieronNan)

def cml(prop, phi, dfs, nombres):
    for i in range(len(dfs)):
        x = np.array([phi(dfs[i][prop].values)]).T
        y = np.array([dfs[i]['precio'].values]).T
        linear_regressor = metnum.LinearRegression()

        linear_regressor.fit(x,y)
        dfs[i]['prediction'] = linear_regressor.predict(x)
        print(nombres[i], ": ", RMSLE(dfs[i]['precio'].values, dfs[i]['prediction'].values))

def cmlConKFold(prop, phi, dfs, nombres, k):
    for i in range(len(dfs)):
        x = np.array([phi(dfs[i][prop].values)]).T
        y = np.array([dfs[i]['precio'].values]).T
        print(nombres[i], ": ", kfold(x, y, k, RMSLE))

def graficarPropXPrecio(prop, propName, dfs, nombres, sigma=16):
    def myplot(x, y, bins=100):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    fig, axs = plt.subplots(1,3, figsize=(20,5))
    for i in range(len(dfs)):
        if(sigma == 0):
            axs[i].scatter(dfs[i][prop], dfs[i]['precio'])
        else:
            img, extent = myplot(dfs[i][prop], dfs[i]['precio'])
            axs[i].imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        axs[i].set_title(nombres[i])
        axs[i].set_ylabel("precio")
        axs[i].set_xlabel(propName)



def oculta_celdas(for_next=False):
    this_cell = """$('div.cell.code_cell.rendered.selected')"""
    next_cell = this_cell + '.next()'

    toggle_text = 'Mostrar/ocultar código de celda'  # text shown on toggle link
    target_cell = this_cell  # target cell to control with toggle
    js_hide_current = ''  # bit of JS to permanently hide code in current cell (only when toggling next cell)

    if for_next:
        target_cell = next_cell
        toggle_text += ' next cell'
        js_hide_current = this_cell + '.find("div.input").hide();'

    js_f_name = 'code_toggle_{}'.format(str(random.randint(1,2**64)))

    html = """
        <script>
            function {f_name}() {{
                {cell_selector}.find('div.input').toggle();
            }}

            {js_hide_current}
        </script>

        <a href="javascript:{f_name}()">{toggle_text}</a>
    """.format(
        f_name=js_f_name,
        cell_selector=target_cell,
        js_hide_current=js_hide_current, 
        toggle_text=toggle_text
    )

    return HTML(html)

#la func contador_fulero te devuelve un pandas.Series con la cantidad de counts sumados de cada palabra
#es medio feo porque podria sumar 3 pero ser todas de la misma palabra.
def contador_fulero(dataF, prop, bagofW):
    j = pd.Series(np.zeros(len(dataF.index)), range(len(dataF.index)))
    for i in range(len(bagofW)):
        j += dataF[prop].str.count(bagofW[i])
    return j


#la func contador_especifico te devuelve un pandas.DataFrame con el count de cada palabra por separado.
def contador_especifico(dataF, prop, bagofW):
    R = pd.DataFrame(index= range(len(dataF.index)))
    for i in range(len(bagofW)):
        R[bagofW[i]] = dataF[prop].str.count(bagofW[i])
    return R

def filterData(data, props, pred = 'precio'):
    return data.dropna(subset=props + [pred])

def predictWithKFold(data, phis, props, pred = 'precio', validation = RMSLE, k = 5):
    data = filterData(data, props, pred)
    b = data[pred]
    data = data[props]
    for i in range(len(props)):
        data[props[i]] = phis[i](data[props[i]])
    data = data.to_numpy()
    b = b.to_numpy().flatten()
    return kfold(data, b, 3, validation)

def graficarRelacionEntrePropiedades(props, data, pred, sigma=16):
    def myplot(x, y, bins=(100,100)):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=sigma)

        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        return heatmap.T, extent

    filas = int(np.ceil(len(props)/3))
    columnas = 3
    fig, axs = plt.subplots(filas,columnas, figsize=(15,15))
    i = 0
    for j in range(filas):
        for k in range(columnas):
            fData = filterData(data, [props[i]], pred)
            if(sigma == 0):
                axs[j][k].scatter(fData[props[i]], fData[pred])
            else:
                img, extent = myplot(fData[props[i]], fData[pred])
                axs[j][k].imshow(img, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
            axs[j][k].set_title(props[i])
            axs[j][k].set_ylabel(pred)
            axs[j][k].set_xlabel(props[i])
            i += 1

def precisionPorGrupos(data, props, groups, propGroup, phis, pred, validation = RMSLE):
    for i in range(len(groups)):
        df_group = data[data[propGroup]==groups[i]]
        val = predictWithKFold(df_group, phis, props, pred=pred, validation = validation)
        print(f"Error de precision para {groups[i]} es de {val}")

def showHeatMap(x, y, xlabel, ylabel, title, sigma=16, bins=(100,100)):
        heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
        heatmap = gaussian_filter(heatmap, sigma=sigma)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        fig, axs = plt.subplots(1,1, figsize=(5,5))
        axs.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.jet, aspect='auto')
        axs.set_title(title)
        axs.set_ylabel(ylabel)
        axs.set_xlabel(xlabel)