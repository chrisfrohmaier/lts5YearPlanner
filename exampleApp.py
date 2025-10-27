import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.express.colors import sample_colorscale
import shapely.geometry
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon 
import json
import numpy as np
from code_editor import code_editor
from io import StringIO
import math
import datetime
import os
from pymongo import MongoClient
import dotenv

def plotEllipseTissot(ra, dec, radius=20):
    theta = np.deg2rad(dec)
    phi = np.deg2rad(ra - 360 if ra > 180 else ra)
    ellipse = Ellipse((phi,theta), 2*np.deg2rad(radius)/ np.cos(theta),
                      2*np.deg2rad(radius))
    vertices = ellipse.get_verts()     # get the vertices from the ellipse object
    
    verticesDeg = np.rad2deg(vertices)
    
    ra_out = [i + 360 if i < 0  else i for i in verticesDeg[:,0]]
    dec_out = verticesDeg[:,1]

    return np.column_stack((ra_out, dec_out))

def makeEllipse(ra_center, dec_center, a,b,theta):
    sample = 20
    alpha = [x / 20.0 * 2.0 * math.pi for x in range(0, sample+1)]
    theta=theta
    x0=ra_center
    y0=dec_center
    x=x0+a*np.sin(theta)
    y=y0+b*np.cos(theta)
    xell=x0+a*np.cos(alpha)*np.cos(theta)-b*np.sin(alpha)*np.sin(theta)
    yell=y0+a*np.cos(alpha)*np.sin(theta)+b*np.sin(alpha)*np.cos(theta)
    return np.column_stack((xell, yell))

f = open('demoArea.json')
io = f.read()

xsize = 420
ysize = xsize/2
longitude = np.linspace(0,360, int(xsize))
latitude = np.linspace(-90, 90, int(ysize))
grid_map_nan = np.load('ltsVPSelfie453.npy')

zmin = 10
zmax = np.nanmax(grid_map_nan)




dataDefault = json.loads(str(io))




st.set_page_config(layout="wide")

st.title("4MOST Year 1 Long Term Scheduler")

st.write("""As part of the 4MOST Long-Term Scheduler development efforts, participating surveys will be able to specifiy a preference for which regions of the sky are observed in year 1.

There is no gauarantee that these regions will be observed to the extent the surveys request. This is simply a method for surveys to define a preference that the scheduler will consider when it attempts to optimse a strategy. Indeed, if all surveys request time, then it will be impossible to satisfy all requests. It is, therefore, also required that surveys provide a scientific justification within their submitted file.

To this end, we have created this web tool for surveys to define their year 1 preference and to submit to the LTS coordinate group. Please follow the process presented below.
""")

st.divider()
st.header("Step 1: Define Areas here")

st.write("""
In this section we will define which regions of the sky we want to observe in year 1.
         
These areas are defined in the JSON editor below, JSON syntax highlighting is also shown. You can either edit the JSON file directly, or paste in your own code. 
         
Click the :grey-background[:orange[Run \u25BA]]. button at the bottom right of the window to process the inputted data. This will update the sky plot in Step 2.
""")

st.markdown("""
Edit the value for the `survey` key with your surveys ID. e.g. S01. Expected format: string.

Enter your science justification for this request in the `scienceJustification` part. Do not use the \'\"\' character as this will break you out of the string.

The `year1Areas` are where you can add your defined polygons, each defined within the array square brackets `[ ]`, add a new element wrapped in `{ }`. There are three polygon types `box`, `circle`, `ellipse` (see below for examples).

#### t_frac
The `t_frac` key is common to all polygons. It is where you define what fraction of the total 5-year observing time you would like to use in Year 1. It should take a value between 0-1.
            
Example Polygons are shown at the bottom of the page.  
""")

#{"name": "Copy", "hasText":True, "alwaysOn": True,"style": {"top": "0.46rem", "right": "0.4rem", "commands": ["copyAll"]}}
custom_btns = [{"name": "Run",
"feather": "Play",
"primary": True,
"alwaysOn":True,
"hasText": True,
"showWithIcon": True,
"commands": ["submit"],
"style": {"bottom": "0.44rem", "right": "0.4rem"}
}]
response_dict = code_editor(str(io), lang="json", buttons=custom_btns, height=[10, 20])

try:
    data = json.loads(str(response_dict['text']))
except:
    data = dataDefault

numColours = np.linspace(0, 1, len(data["year1Areas"])+1)
colours = iter(sample_colorscale('Tealgrn', list(numColours)))

def colorbar(zmin, zmax, n = 6):
    return dict(
        title = "Total Exposure Time<br>in pixel (minutes)",
        tickmode = "array",
        tickvals = np.linspace(np.log10(zmin), np.log10(zmax), n),
        ticktext = np.round(10 ** np.linspace(np.log10(zmin), np.log10(zmax), n), 0)
    )





layout = go.Layout(
    autosize=False,
    width=800, 
    height=600,
    title='Year 1 Long Term Scheduler Preference: SELFIE 453',
    xaxis=dict(
        title='R.A.',

    ),
    yaxis=dict(
        title='Declination',

    ))


fig = go.Figure(go.Heatmap(
        x=longitude,
        y=latitude,
        z=np.ma.log10(grid_map_nan),
    text=grid_map_nan,
    hovertemplate = 
    "<i>4MOST VP Exposure Time</i><br>" +
    "<b>RA</b>: %{x}<br>" +
    "<b>Decl.</b>: %{y}<br>" +
    "<b>Total t_exp (min)</b>: %{text:.1f}",
    zmin = np.log10(zmin), zmax = np.log10(zmax),
    colorbar = colorbar(zmin, zmax, 12),
    colorscale = 'Plasma',
    name=""
    ), layout=layout)

for i in data["year1Areas"]:

    if i['type']=='box':
        RA = i['RA']
        Dec = i['Dec']
        tfrac = i['t_frac']
        convex_hull = np.array(
        shapely.geometry.MultiPoint(
            [xy for xy in zip(RA, Dec)]
        ).convex_hull.exterior.coords
        )
        fig.add_trace(go.Scatter(
            x=convex_hull[:, 0],
            y=convex_hull[:, 1],
            showlegend=False,
            mode="lines",
            fill="toself",
            line=dict(color=next(colours), width=2),
            name="t_frac: "+str(tfrac)
        )
                     )
    elif i['type']=='circle':
        ra_center = i['RA_center']
        dec_center = i['Dec_center']
        radius = i['radius']
        tfrac = i['t_frac']
        tissot = plotEllipseTissot(ra_center, dec_center, radius = radius)
        fig.add_trace(go.Scatter(
        x=tissot[:, 0],
        y=tissot[:, 1],
        showlegend=False,
        mode="lines",
        fill="toself",
        line=dict(color=next(colours), width=2),
        name="t_frac: "+str(tfrac)
    )
                 )
    
    elif i['type']=='ellipse':
        tfrac = i['t_frac']
        ellipse = makeEllipse(i['RA_center'], i['Dec_center'], i['a'],i['b'],np.deg2rad(i['theta']) )
        ellipse_convex_hull = np.array(
        shapely.geometry.MultiPoint(
            ellipse
        ).convex_hull.exterior.coords
        )
        fig.add_trace(go.Scatter(
        x=ellipse_convex_hull[:, 0],
        y=ellipse_convex_hull[:, 1],
        showlegend=False,
        mode="lines",
        fill="toself",
        line=dict(color=next(colours), width=2),
        name="t_frac: "+str(tfrac)
    )
                 )
        
    else:
        print("Please enter a valid shape: 'box', 'circle', 'ellispe'")
        continue

fig['layout']['xaxis']['autorange'] = "reversed"
fig.update_layout(yaxis_range=[-90,30])

st.divider()
st.header("Step 2: Check output on sky map")
st.markdown("""
Inspect your polygons here. If you are happy with the result, move on to the next step.
""")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.header("Step 3: Download JSON file")
st.markdown("""
Whatever is displayed in the above plot will be downloaded in this section.
            
Select your survey from the dropdown list, click the download button.
""")
sb = st.columns((1,9))
surveyNumber=None
surveyNumber = sb[0].selectbox(
    'Select Survey',
    ('01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18'),
    index=None,
    placeholder="S00")
if surveyNumber == None:
    surveyNumber = '00'
today = datetime.date.today()
fileOutputName = 'S'+str(surveyNumber)+'_'+'LTSYear1'+'_'+str(today.year)+today.strftime('%m')+today.strftime('%d')+'.json'
st.write('File name:', fileOutputName)
json_string = json.dumps(data,indent=4, separators=(',', ': '))

def save_to_remote_db(json_text, filename):
    """
    Save JSON to a MongoDB collection.
    Expects environment variables (or defaults):
      MONGO_URI          - MongoDB connection string (required)
      MONGO_DB           - database name (default: 'lts')
      MONGO_COLLECTION   - collection name (default: 'year1submissions')
    The inserted document will have fields:
      filename, timestamp (UTC ISO), data (parsed JSON)
    """
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        return False, "MONGO_URI environment variable not set."

    db_name = os.environ.get("MONGO_DB", "lts")
    coll_name = os.environ.get("MONGO_COLLECTION", "year1submissions")

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # trigger connection check
        client.server_info()

        db = client[db_name]
        coll = db[coll_name]

        doc = {
            "filename": filename,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "data": json.loads(json_text)
        }

        result = coll.insert_one(doc)
        client.close()
        return True, f"Saved to MongoDB (id: {result.inserted_id})."
    except Exception as e:
        try:
            client.close()
        except:
            pass
        return False, f"Error saving to MongoDB: {e}"

# Replace single download button with side-by-side download + save
# ...existing code...
cols = st.columns((1,1))
with cols[0]:
    st.download_button(
        label="Download JSON File",
        data=json_string,
        file_name=fileOutputName,
        mime="application/json",
    )
with cols[1]:
    if st.button("Save to remote DB"):
        ok, msg = save_to_remote_db(json_string, fileOutputName)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

st.divider()
st.header("Example JSON inputs")
st.markdown("""Here are three example polygons you can copy, paste, and edit!

All units are in degrees.
""")

c1, c2, c3 = st.columns((1, 1, 1))
c1.header("Polygon")
c1.json(      {
        "name": "examplePolygon",
        "type": "box",
        "RA": [0.0 ,52.5, 52.5, 0.0],
        "Dec":[-35.0 ,-35.0,-25.0,-25.0],
        "t_frac": 0.2
      })
c2.header('Circle')
c2.json({
        "name": "exampleCircle",
        "type": "circle",
        "RA_center": 200,
        "Dec_center":0,
        "radius": 5,
        "t_frac": 0.6
      })
c3.header('Ellipse')
c3.json(      {
        "name": "exampleEllipse",
        "type": "ellipse",
        "RA_center": 283.8313,
        "Dec_center":-30.5453,
        "a": 13.0,
        "b":4.5,
        "theta": -11.5,
        "t_frac": 0.6
      })