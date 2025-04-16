import polars as pl
import matplotlib.pyplot as plt

class iris2D:
    def __init__(self):
        """
        This is an example of a good discretization of the iris dataset in 2D.
        It has 1 non-deterministic pair in the original data.
        Cuts at:

        - X1: 4.95, 5.85
        - X2: 2.85

        Non-deterministic pairs: 1 (at x1=4.9, x2=2.5)
        """
        self.disc_df = pl.read_csv('DISCiris2D.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'Dec'])
        self.data_df = pl.read_csv('iris2D.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'Dec'])

        self.x1_cut = [4.95, 5.85]
        self.x2_cut = [2.85]
        self.title = 'Iris Dataset 2D Good Discretization'
    
    def plot(self):

        plt.scatter(self.data_df[:, 0], self.data_df[:, 1], c=self.data_df[:, 2], cmap='viridis')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title(self.title)
        plt.colorbar(label='Dec')
        for i in range(len(self.x1_cut)):
            plt.axvline(x=self.x1_cut[i], color='#FFC0CB', linestyle='--')
        for i in range(len(self.x2_cut)):
            plt.axhline(y=self.x2_cut[i], color='#5CE65C', linestyle='--')
        plt.show()

class iris3D:
    def __init__(self):
        """
        This is an example of a good discretization of the iris dataset in 3D.
        Cuts at:

        - X1: 8.5
        - X2: 2.45, 2.65
        - X3: 5.0

        Non-deterministic pairs: 0
        """
        self.data_df = pl.read_csv('iris3D.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'x3', 'Dec'])
        self.disc_df = pl.read_csv('DISCiris3D.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'x3', 'Dec'])
        self.x3_cut = [5.0]
        self.x2_cut = [2.45, 2.65]
        self.x1_cut = [8.5]

        self.title = 'Iris Dataset 3D Good Discretization'

    def plot(self):
        fig = plt.figure(figsize=(18, 12))

        # 3D scatter plot
        ax1 = fig.add_subplot(221, projection='3d')
        ax1.scatter(self.data_df['x1'], self.data_df['x2'], self.data_df['x3'], c=self.data_df['Dec'], cmap='viridis')
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('x3')
        ax1.set_title(self.title)

        # View from top
        ax2 = fig.add_subplot(222)
        ax2.scatter(self.data_df['x1'], self.data_df['x2'], c=self.data_df['Dec'], cmap='viridis')
        ax2.set_xlabel('x1')
        for i in range(len(self.x1_cut)):
            ax2.axvline(x=self.x1_cut[i], color='#154C79', linestyle='--')
        ax2.set_ylabel('x2')
        for i in range(len(self.x2_cut)):
            ax2.axhline(y=self.x2_cut[i], color='#5CE65C', linestyle='--')
        ax2.set_title('View from Top')

        # View from front
        ax3 = fig.add_subplot(223)
        ax3.scatter(self.data_df['x2'], self.data_df['x3'], c=self.data_df['Dec'], cmap='viridis')
        ax3.set_xlabel('x2')
        for i in range(len(self.x2_cut)):
            ax3.axvline(x=self.x2_cut[i], color='#5CE65C', linestyle='--')
        ax3.set_ylabel('x3')
        for i in range(len(self.x3_cut)):
            ax3.axhline(y=self.x3_cut[i], color='#FFC0CB', linestyle='--')
        ax3.set_title('View from Front')

        # View from side
        ax4 = fig.add_subplot(224)
        ax4.scatter(self.data_df['x1'], self.data_df['x3'], c=self.data_df['Dec'], cmap='viridis')
        ax4.set_xlabel('x1')
        for i in range(len(self.x1_cut)):
            ax4.axvline(x=self.x1_cut[i], color='#154C79', linestyle='--')
        ax4.set_ylabel('x3')
        for i in range(len(self.x3_cut)):
            ax4.axhline(y=self.x3_cut[i], color='#FFC0CB', linestyle='--')
        ax4.set_title('View from Side')

        plt.tight_layout()
        plt.show()

class iris3DBAD(iris3D):
    def __init__(self):
        """
        This is an example where there are no non-deterministic pairs in the original data.
        But due to poor discretization, the discretized data now has 1 non-deterministic pair.
        Cuts at:

        - X1: No cuts
        - X2: 2.45, 2.65
        - X3: 5.0

        Non-deterministic pairs: 0 in original data, 1 in discretized data
        """
        self.data_df = pl.read_csv('iris3D.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'x3', 'Dec'])
        self.disc_df = pl.read_csv('DISCiris3DBAD.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'x3', 'Dec'])
        self.x3_cut = [5]
        self.x2_cut = [2.45, 2.65]
        self.x1_cut = []

        self.title = 'Iris Dataset 3D Bad Discretization'

class NoDecisionRange(iris2D):
    def __init__(self):
        """
        This is an example of data where there is one sector left without a decision range.
        If we were to put a point at x1=2.0 x2=0 there would be no decision.
        Cuts at:

        - X1: 1.5
        - X2: 0.5, 1.5

        Non-deterministic pairs: 1 (at x1=4.9, x2=2.5)
        """
        self.disc_df = pl.read_csv('DISCnodec.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'Dec'])
        self.data_df = pl.read_csv('nodec.csv', separator=',', has_header=False, new_columns=['x1', 'x2', 'Dec'])

        self.x1_cut = [1.5]
        self.x2_cut = [0.5, 1.5]
        self.title = 'No Decision Range'

if __name__ == "__main__":
    iris = iris2D()
    iris.plot()
    iris = iris3D()
    iris.plot()
    iris = iris3DBAD()
    iris.plot()
    nodec = NoDecisionRange()
    nodec.plot()