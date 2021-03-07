# Imports
import numpy as np
import pandas as pd

import bokeh
from bokeh.io import output_notebook
from bokeh.models import CheckboxButtonGroup, RangeSlider, Spinner, CustomJS, ColumnDataSource, Button, Circle, LinearColorMapper, PreText
from bokeh.models.widgets import Slider
from bokeh.layouts import column, row
from bokeh.plotting import show,figure
from bokeh.application import Application
from bokeh.application.handlers import FunctionHandler
from bokeh.models.renderers import GlyphRenderer
from bokeh.models.tools import HoverTool

from datetime import datetime
from tabulate import tabulate
from typing import Optional

from .dataset import Dataset
from .stream_parser import StreamParser



def bokehview(  ds: Optional[Dataset] = None, stream: Optional[StreamParser] = None, 
                initial_shot = 0, initial_peak_size = 10, initial_pred_size = 10,
                minI = 0, maxI = 1000, colormap = 'Inferno256',
                columns_for_display: list = ['Event', 'Event_raw',
                                            'run','region', 'sample', 'crystal_id',
                                            'file', 'file_raw',
                                            'com_x', 'com_y','center_refine_score', 'center_x', 'center_y',
                                            'selected'],
                peak_glyph_type = 'circle', pred_glyph_type = 'square',
                layout_max_width: int = 1300,
                detector_size = (2048,2048)):

    #Set bokeh for notebook  
    output_notebook()

    if (ds is None)&(stream is None):
        raise ValueError('A Dataset object and/or a StreamParser object must be given as input')
    elif ds is not None:
        Nshot,xsize,ysize = ds.stacks['corrected'].shape
    elif stream is not None:
        Nshot = stream.shots.shape[0]
        xsize,ysize = detector_size

    #Color map
    mapper = LinearColorMapper(
        palette=colormap,
        low=minI,
        high=maxI
    )

    #Widgets
    ##Sliders
    Shot_slider = Slider(title="Shot", value=initial_shot, start=0, end=(Nshot-1), step=1, sizing_mode="scale_width")
    Intensity_range = RangeSlider(start=minI, end=maxI, value=(minI,maxI), step=1, title="Intensity", sizing_mode="scale_width")
    ##Buttons
    Options = CheckboxButtonGroup(labels=["Data", "Peaks", "Predictions"], active=[],sizing_mode="scale_width")
    Select_button = Button(label='Select pattern',button_type="success", sizing_mode="scale_width",name = 'select_button')
    ##Spinners
    Peak_glyph_size = Spinner(title="Peak glyphs", low=0, high=40, step=0.5, value=initial_peak_size, sizing_mode="fixed", width = 80)
    Preds_glyph_size = Spinner(title="Prediction glyphs", low=0, high=40, step=0.5, value=initial_peak_size, sizing_mode="fixed", width = 110)
    Shot_spinner = Spinner(title="Shot", low=0, high=Nshot, step=1, value=initial_shot, sizing_mode="fixed", width = 80)
    ##Text
    stats = PreText(text='', width=500)

    #Graph 
    p = figure(tools = 'box_zoom,wheel_zoom,pan,reset,save')

    source_peaks ={'x':np.array([0]), 'y':np.array([0])}
    dummy_peaks = p.scatter(x='x', y='y', source=source_peaks, size=initial_peak_size, name='peaks', line_color="white", fill_color="darkgreen", marker=peak_glyph_type, alpha=0.5)#
    dummy_peaks.visible = False

    source_preds ={'x':np.array([0]), 'y':np.array([0])}
    dummy_preds = p.scatter(x='x', y='y', source=source_preds, size=initial_peak_size, name='predictions', line_color="navy", fill_color="skyblue", marker=pred_glyph_type, alpha=0.3)
    dummy_preds.visible = False

    p.x_range.range_padding = p.y_range.range_padding = 0
    p.grid.grid_line_width = 0

    p.toolbar.autohide = True
    p.xaxis.bounds = (0,xsize)
    p.yaxis.bounds = (0,ysize)

    #Interactions
    ##Updates text panel
    def update_stats(series):
        text = []
        for var,val in series.iteritems():
            text.append([var,val])
        stats.text = tabulate(text)

    ##Updates 'selected' column of the dataset
    def update_select(ev):
        shot = Shot_spinner.value
        if ds is not None:
            ds.shots.loc[shot,'selected'] = np.invert(ds.shots.loc[shot,'selected'])
            update_stats(ds.shots[columns_for_display].loc[shot])
    
    ##Updates displayed pattern
    def update(attr, old, new):
            
        #Parameters
        shot = Shot_spinner.value    
        opt = Options.active
        peak_size = Peak_glyph_size.value
        pred_size = Preds_glyph_size.value
        IRange = Intensity_range.value
        
        #Makes sure that shots correspond in dataset and stream
        if (ds is not None)&(stream is not None): 
            ds_shotid = ds.shots.loc[shot][['region','crystal_id', 'run', 'sample']]
            try:
                stream_id = stream.shots.rename(columns={'hdf5/%/shots/region':'hdf5_region',
                                                    'hdf5/%/shots/crystal_id':'hdf5_crystal_id',
                                                    'hdf5/%/shots/run':'hdf5_run',
                                                    'hdf5/%/shots/sample':'hdf5_sample'}
                                            ).query(f'hdf5_region == {ds_shotid.region}'
                                                    ).query(f'hdf5_crystal_id == {ds_shotid.crystal_id}'
                                                            ).query(f'hdf5_run == {ds_shotid.run}'
                                                                    ).query('hdf5_sample == "' + ds.shots.loc[shot,"sample"] + '"'
                                                                            )
            except ValueError:
                print("The metadata of this shot is not consistent between dataset and stream")
        else:
            stream_id = None
        
        #Update color map
        mapper.low = IRange[0]
        mapper.high = IRange[1]
        
        #Displayed data
        if ds is not None:
            update_stats(ds.shots[columns_for_display].loc[shot])
            
        #Patterns
        if 0 in opt:  #If the "pattern" option is selected
            
            if ds is not None:
                #Update frame
                frame = ds.stacks['corrected'][shot,:,:].compute()

                if p.select('pattern') != []:
                    pattern_handle = p.select('pattern')
                    pattern_handle.data_source.data['image'] = [frame]
                else:
                    p.image(image=[frame], x=0, y=0, dw=frame.shape[0], dh=frame.shape[0], color_mapper=mapper, level="image", name = 'pattern')
                    hover = HoverTool(names=['pattern'], tooltips=[("I", "@image"),("x", "$x"),("y", "$y")])
                    p.add_tools(hover)
        
        else:        #If the "pattern" option is deselected
            if ds is not None:
                if p.select('pattern') != []:
                    pattern_handle = p.select('pattern')
                    pattern_handle.data_source.data['image'] = []
            
        #Peaks
        if 1 in opt: #If the "peaks" option is selected
            
            if ds is not None:
                #fetch data
                n = ds.stacks['nPeaks'][shot].compute()
                x = ds.stacks['peakXPosRaw'][shot,:n].compute()
                y = ds.stacks['peakYPosRaw'][shot,:n].compute()
                source_peaks = {'x':x, 'y':y}
                
            elif stream is not None:
                x = stream.peaks.query('serial == @shot')['fs/px'].values
                y = stream.peaks.query('serial == @shot')['ss/px'].values
                source_peaks = {'x':x, 'y':y}
                        
            
            if (ds is not None)|(stream is not None):
                if p.select('peaks') != []:
                    peaks_handle = p.select('peaks')
                    peaks_handle.data_source.data = source_peaks
                    peaks_handle.visible = True

                    #Update glyph size
                    peaks_handle.glyph.size = peak_size      
        
        else:        #If the "peaks" option is deselected
            if (ds is not None)|(stream is not None):
                if p.select('peaks') != []:
                    peaks_handle = p.select('peaks')
                    source_peaks = {'x':np.array([0]), 'y':np.array([0])}
                    peaks_handle.data_source.data = source_peaks
                    peaks_handle.visible = False
            
        #Predictions
        if 2 in opt: #If the "predictions" option is selected
            
            if stream is not None:
                #Update peaks
                if stream_id is None:
                    x = stream.indexed.query('serial == @shot')['fs/px'].values
                    y = stream.indexed.query('serial == @shot')['ss/px'].values
                elif stream_id is not None:
                    x = stream.indexed.query(f'serial == {stream_id.serial.values[0]}')['fs/px'].values
                    y = stream.indexed.query(f'serial == {stream_id.serial.values[0]}')['ss/px'].values
                    
                source_preds = {'x':x, 'y':y}

                if p.select('predictions') != []:
                    preds_handle = p.select('predictions')
                    preds_handle.data_source.data = source_preds
                    preds_handle.visible = True

                    #Update glyph size
                    preds_handle.glyph.size = pred_size
                    
                
        else:        #If the "predictions" option is deselected
            if stream is not None:
                if p.select('predictions') != []:
                    preds_handle = p.select('predictions')
                    source_preds = {'x':np.array([0]), 'y':np.array([0])}
                    preds_handle.data_source.data = source_preds
                    preds_handle.visible = False


    Peak_glyph_size.on_change('value_throttled', update)
    Preds_glyph_size.on_change('value_throttled', update)
    Shot_spinner.on_change('value_throttled',update)
    Shot_slider.on_change('value_throttled',update)
    Shot_spinner.js_link('value', Shot_slider, 'value')
    Shot_slider.js_link('value', Shot_spinner, 'value')
    Intensity_range.on_change('value_throttled',update)
    Options.on_change('active',update)
    Select_button.on_click(update_select)

    #layout
    if ds is not None:
        layout = row(  column( row( Options ), row( column( Shot_spinner ), column( Peak_glyph_size ), column( Preds_glyph_size ) ), row( Shot_slider ), row( Intensity_range ), row( Select_button ), row(stats), sizing_mode="fixed", width = 500), column(p, sizing_mode="scale_both"), sizing_mode="scale_both")
    else:
        layout = row(  column( row( Options ), row( column( Shot_spinner ), column( Peak_glyph_size ), column( Preds_glyph_size ) ), row( Shot_slider ), row( Intensity_range ), sizing_mode="fixed", width = 500), column(p, sizing_mode="scale_both"), sizing_mode="scale_both")


    #Required for interactivity
    def modify_doc(doc):
        doc.add_root(row(layout,max_width=layout_max_width))
        doc.title = "Dataset viewer"

    handler = FunctionHandler(modify_doc)
    app = Application(handler)

    #Show
    show(app)