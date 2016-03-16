from bokeh.plotting import figure, output_server, show, curdoc
from bokeh.models import ColumnDataSource
import numpy as np
import time

# def plot(validation_result, epochs):
#   p = figure(background_fill_color='#F0E8E2', title="Learning curve")
#   p.xgrid.grid_line_color = 'white'
#   p.ygrid.grid_line_color = 'white'
#   p.xaxis.axis_label = 'Epoch'
#   p.yaxis.axis_label = 'Correct guesses'

#   # ds = ColumnDataSource(data=dict(x=[], y=[]))

#   # s = ColumnDataSource(data=dict(x=np.arange(epochs), y=validation_result))
#   r = p.line(x=[], y=[], line_width=2)
#   ds = r.data_source

#   def callback(attr,old,new):
#       ds.data['x'].append(validation_result[i])
#       ds.data['y'].append(validation_result[i])
#       ds.trigger('data', ds.data, ds.data)

#   def changedata():
#       global i
#       for j in range(4):
#           i += 1
#           ds.data['x'].append(validation_result[i])
#           ds.data['y'].append(validation_result[i])
#           time.sleep(2)

#   changedata()
#   ds.on_change('data',callback)
#   curdoc().add_root(p)

class Net:
    def __init__(self):
        self.res = []
        
    def plot(self, epochs):
        global s
        p = figure(background_fill_color='#F0E8E2', title="Learning curve")
        s = ColumnDataSource(data=dict(x=self.res, y=self.res))
        # s1 = ColumnDataSource(data=dict(x=[]], y=[]])
        # p.line('x','y', source=s) # list data
        # p1.line('x','y', source=s1) # empty


        r = p.line(x=[],y=[], line_width=2)
        ds = r.data_source

        def callback(attr,old,new):
            global plotiter
            ds.data['x'].append(self.res[plotiter])
            ds.data['y'].append(self.res[plotiter])
            ds.trigger('data', ds.data, ds.data)
            plotiter += 1
            print "whats upppppp"

        s.on_change('data',callback)
        curdoc().add_root(p)

    def changedata(self):
        global s
        print s
        self.res.append(5)
        self.res.append(32)
        self.res.append(6)
        self.res.append(87)
        print self.res
    
s = 0
net = Net()
plotiter = 0
# k = [43,9465,546,9253,9623]
net.plot(30)
print net.res
time.sleep(3)
net.changedata()




