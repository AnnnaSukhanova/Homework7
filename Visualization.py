import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px


class Visualization:

    def __create_df(self, data, columns):
        return pd.DataFrame(data=data, columns=columns)

    def models_error_scatter_plot(self, accuracy_valid, accuracy_test, title, show=False, save=False, name=None,
                                  path2save=None):
        """
        :param error_valid: numpy.array - errors on validation set
        :param error_test: numpy.array - errors on test set
        :param names: numpy.array of strings - how y(x,w) looks
        :param lambda_lst: numpy.array - lambda value
        :param title: title of plot
        :param show: (bool) optional if True show figure in browser
        :param save: (bool) optional if True save figure in html format
        :param name: (str) optional name of html file
        :param path2save: (str) optional path to directory, where html file is going to be saved
        example
            /dir/dir/
        """
        df = self.__create_df(np.stack((accuracy_valid,
                                accuracy_test), axis=1),
                              ['accuracy_valid', 'accuracy_test'])
        fig = px.scatter(df, y="accuracy_valid", x="accuracy_test")

        fig.update_layout(title=title)
        if show:
            fig.show()
        if save:
            assert name is not None, "name shouldn't be None if  save is True"
            if path2save:
                fig.write_html(f"{path2save}/{name}.html")
            else:
                fig.write_html(f"{name}.html")