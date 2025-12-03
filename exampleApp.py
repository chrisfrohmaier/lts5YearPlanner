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
from datetime import timezone

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
# new helper: convert axis-aligned bounds to rectangle corner coordinates
def rect_corners(xmin, xmax, ymin, ymax, closed=False):
    """
    Return corner coordinates for an axis-aligned rectangle defined by bounds.
    Parameters:
      xmin, xmax, ymin, ymax : numeric
      closed (bool) : if True, repeat the first corner at the end (useful for plotting closed polygons)
    Returns:
      numpy.ndarray shape (4,2) or (5,2) if closed: [[x1,y1], [x2,y2], [x3,y3], [x4,y4], ...]
    Order of corners: (xmin,ymin), (xmax,ymin), (xmax,ymax), (xmin,ymax)
    """
    corners = np.array([
        [xmin, ymin],
        [xmax, ymin],
        [xmax, ymax],
        [xmin, ymax],
    ], dtype=float)
    if closed:
        return np.vstack([corners, corners[0]])
    return corners


def plotPolygons(data, survey_id, allColours=True):
    if allColours==True:
        numColours = np.linspace(0, 1, len(data["year1Areas"])+1)
        colours = iter(sample_colorscale('Tealgrn', list(numColours)))
    else:
        numColours = np.linspace(0, 1, len(data["year1Areas"])+1)
        colours = iter(sample_colorscale('Turbo', list(numColours)))
    for i in data["year1Areas"]:

        if i['type']=='stripe':
            RA_lower = i['RA_lower']; RA_upper = i['RA_upper']
            Dec_lower = i['Dec_lower']; Dec_upper = i['Dec_upper']
            tfrac = i['t_frac']
            # compute RA span, handle wrap-around across 360->0
            if RA_upper >= RA_lower:
                ra_span = RA_upper - RA_lower
            else:
                ra_span = (RA_upper + 360.0) - RA_lower
            dec_span = abs(Dec_upper - Dec_lower)
            # enforce minimum span of 2.5 degrees for both axes
            min_span = 2.5
            if dec_span < min_span:
                st.warning(
                    f"Stripe '{i.get('name','')}' for survey {survey_id} is too small: "
                    f"Dec span={dec_span:.2f}. Minimum is {min_span}.\n This will not be plotted and is not a valid LTS input."
                )
                continue
            # build rectangle corners and plot
            corners = rect_corners(RA_lower, RA_upper, Dec_lower, Dec_upper, closed=True)
            fig.add_trace(go.Scatter(
                x=corners[:, 0],
                y=corners[:, 1],
                showlegend=False,
                mode="lines",
                fill="toself",
                line=dict(color=next(colours), width=2),
                name=f"{survey_id}<br> t_frac: {tfrac}"
            )
                        )
        elif i['type']=='point':
            ra_center = i['RA_center']
            dec_center = i['Dec_center']
            radius = 1.15
            tfrac = i['t_frac']
            tissot = plotEllipseTissot(ra_center, dec_center, radius = radius)
            fig.add_trace(go.Scatter(
            x=tissot[:, 0],
            y=tissot[:, 1],
            showlegend=False,
            mode="lines",
            fill="toself",
            line=dict(color=next(colours), width=2),
            name=f"{survey_id}<br> t_frac: {tfrac}"
        )
                    )
        
            
        else:
            print("Please enter a valid shape: 'stripe', 'point'")
            continue


def computeTimePressures(data):
    truthGrids = []
    for i in data["year1Areas"]:

        if i['type']=='stripe':
            RA_lower = i['RA_lower']; RA_upper = i['RA_upper']
            Dec_lower = i['Dec_lower']; Dec_upper = i['Dec_upper']
            tfrac = i['t_frac']
            convex_hull = np.array(
                shapely.geometry.MultiPoint(
                    rect_corners(RA_lower, RA_upper, Dec_lower, Dec_upper, closed=True)
                ).convex_hull.exterior.coords
                )

        elif i['type']=='point':
            ra_center = i['RA_center']
            dec_center = i['Dec_center']
            radius = 1.15
            tfrac = i['t_frac']
            tissot = plotEllipseTissot(ra_center, dec_center, radius = radius)
            convex_hull = np.array(
                shapely.geometry.MultiPoint(
                    tissot
                ).convex_hull.exterior.coords
                )
        poly = Polygon(convex_hull)
        allPoints = np.vstack(list(map(np.ravel, mesh))).T
        sPoints = shapely.points(allPoints)
        inShape = poly.contains(sPoints)
        weightMapFlat = inShape * tfrac
        weightMap = np.reshape(weightMapFlat, grid_map_nan.shape)
        truthGrids.append(weightMap)
    truthGrid = np.maximum.reduce(truthGrids)
    return truthGrid

def moving_average_2d_wrap(arr, width):
    # width must be odd so the window is centered
    assert width % 2 == 1, "width must be odd"

    k = width // 2
    result = np.zeros_like(arr, dtype=float)

    for dx in range(-k, k + 1):
        for dy in range(-k, k + 1):
            result += np.roll(np.roll(arr, dx, axis=0), dy, axis=0)

    return result / (width * width)

# new helper: 1D circular moving average (for longitude series)
def moving_average_1d_wrap(arr, width):
    """
    Circular moving average over 1D array.
    width must be odd. Returns array same shape as input.
    """
    arr = np.asarray(arr, dtype=float)
    assert width % 2 == 1, "width must be odd"
    k = width // 2
    # use rolling sum via np.roll
    result = np.zeros_like(arr, dtype=float)
    for shift in range(-k, k+1):
        result += np.roll(arr, shift)
    return result / width


f = open('demoArea.json')
io = f.read()

xsize = 420
ysize = xsize/2
longitude = np.linspace(0,360, int(xsize))
latitude = np.linspace(-90, 90, int(ysize))
mesh = np.meshgrid(longitude, latitude)
grid_map_nan = np.load('ltsVPSelfie453.npy')

zmin = 10
zmax = np.nanmax(grid_map_nan)




dataDefault = json.loads(str(io))




# new: helpers to parse ISO timestamps and fetch latest submission per survey
def _parse_iso_ts(ts):
    import datetime as _dt
    if isinstance(ts, _dt.datetime):
        return ts
    try:
        # convert trailing 'Z' to +00:00 so fromisoformat accepts it
        return _dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None

def get_latest_submissions_by_survey(mongo_uri, db_name="lts", coll_name="year1submissions"):
    """
    Return dict: { survey_id: { "data": <doc.data>, "timestamp": <doc.timestamp>, "filename": <doc.filename>, "_id": <doc._id> } }
    where the document chosen is the latest (by timestamp) for that survey.
    """
    client = None
    latest = {}
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.server_info()  # force connect
        coll = client[db_name][coll_name]
        for doc in coll.find({}):
            survey = None
            try:
                survey = doc.get("data", {}).get("survey")
            except Exception:
                survey = None
            if not survey:
                continue
            ts = _parse_iso_ts(doc.get("timestamp"))
            if ts is None:
                continue
            cur = latest.get(survey)
            if cur is None or ts > cur["_parsed_ts"]:
                # store parsed timestamp for comparison and keep full doc
                latest[survey] = {"_parsed_ts": ts, "doc": doc}
        # build return mapping that exposes the full "data" column plus metadata
        result = {}
        for s, entry in latest.items():
            d = entry["doc"]
            result[s] = {
                "data": d.get("data"),
                "timestamp": d.get("timestamp"),
                "filename": d.get("filename"),
                "_id": d.get("_id")
            }
        return result
    except Exception as e:
        # on error return empty dict (caller may show message)
        return {}
    finally:
        try:
            if client:
                client.close()
        except Exception:
            pass

st.set_page_config(layout="wide")

# query MongoDB for latest submissions per survey (if secrets provided)
try:
    mongo_uri = st.secrets.get("MONGO_URI")
    mongo_db = st.secrets.get("MONGO_DB", "lts")
    mongo_coll = st.secrets.get("MONGO_COLLECTION", "year1submissions")
except Exception:
    mongo_uri = None
    mongo_db = "lts"
    mongo_coll = "year1submissions"

latest_submissions = {}
if mongo_uri:
    latest_submissions = get_latest_submissions_by_survey(mongo_uri, mongo_db, mongo_coll)

# convert latest_submissions to a JSON string for download/display (ObjectId/datetime -> str)
try:
    latest_submissions_json = json.dumps(latest_submissions, indent=4, default=str)
except Exception:
    print("Error converting latest submissions to JSON")
    latest_submissions_json = "{}"

#print(latest_submissions)
# show a compact summary in the sidebar
# if latest_submissions:
#     try:
#         # display full contents for each survey
#         st.sidebar.header("Latest submissions by survey")
#         st.sidebar.json(latest_submissions)
#     except Exception:
#         pass

st.title("4MOST Year 1 Long Term Scheduler V2")

st.write("""This web tool allows 4MOST surveys to define their Year 1 sky observation preferences as part of the Long-Term Scheduler development efforts.
""")

st.divider()
st.header("Step 1: Define Areas here")

st.write("""
Only stripes and single pointings are accepted into the 4MOST Year 1 Long Term Scheduler V2.
         
Edit the JSON contents below, or paste in your own code. 
         
Click the :grey-background[:orange[Run \u25BA]] button. when ready to view submission.
Both the Sky Plot and R.A. Pressure will update accordingly.
""")

st.markdown("""
Edit the value for the `survey` key with your surveys ID. e.g. S01, S02, etc. Expected format: string.

**Important:** Enter your science justification for this request in the `scienceJustification` part. In the event of oversubscription in an R.A. range, the science justification is a useful negotiation tool.
 Do not use the \'\"\' character as this will break you out of the string.

Only acceptable inputs for polygons are: 'stripe' and 'point'. Examples are shown at the end of the page.

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

truthGridCurrent = computeTimePressures(data)
#print(truthGridCurrent)
# numColours = np.linspace(0, 1, len(data["year1Areas"])+1)
# colours = iter(sample_colorscale('Tealgrn', list(numColours)))

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

if latest_submissions:
    for i in latest_submissions.keys():
        dataLatest = latest_submissions[i]['data']
        plotPolygons(dataLatest, dataLatest.get('survey', i), allColours=False)
else:
    st.info("No previous submissions found in the remote DB — only the current edited data will be plotted.")

plotPolygons(data, data['survey'], allColours=True)

if latest_submissions:
    latestTPress = []
    for i in latest_submissions.keys():
        dataLatest = latest_submissions[i]['data']
        latestTPress.append(computeTimePressures(dataLatest))
    truthGridLatest = np.maximum.reduce(latestTPress)
try:
    #print(truthGridLatest)
    truthGridLatest = np.maximum.reduce([truthGridCurrent, truthGridLatest])
    scaledGrid = (truthGridLatest) * grid_map_nan


    time5year = np.nansum(grid_map_nan, axis=0)
    timeMax1year = time5year/5.0
    timeY1 = np.nansum(scaledGrid, axis=0)

    widthWant = len(longitude)/360
    binsWant = 30//widthWant
    coarseTime = timeY1/timeMax1year
    smoothTime = moving_average_2d_wrap(coarseTime, width=25)
    #print(smoothTime)
    plotSmooth = True
except:
    plotSmooth = False

fig['layout']['xaxis']['autorange'] = "reversed"
fig.update_layout(yaxis_range=[-90,30])

st.divider()
st.header("Step 2: Check output on sky map")
st.markdown("""
Inspect the sky map here before moving on to the submission step.

The goal is to avoid oversubscription in Year 1 any R.A. range, which is indicated by the R.A. Time Pressure plot below the sky map.
We do not want to spend more than 50% of the available time in any R.A. range.
R.A. pressure is smoothed over a rolling 30 degrees width.
""")
st.plotly_chart(fig, use_container_width=True)

# New: 1D line plot under the map that shares the longitude x-axis scale
#import plotly.graph_objects as go as _go  # avoid name clash in context; use existing go normally
if plotSmooth:
    fig_times = go.Figure()
    fig_times.add_trace(go.Scatter(
        x=longitude,
        y=coarseTime,
        mode="lines",
        name="Coarse Bins",
        line=dict(color="#b1b1b1", width=2, dash='dash')
    ))
    fig_times.add_trace(go.Scatter(
        x=longitude,
        y=smoothTime,
        mode="lines",
        name="30 degree Smoothing",
        line=dict(color="#96cefd", width=5)
    ))
    fig_times.add_hline(y=0.5, line_width=2, line_dash="dash", line_color="#72e06a", annotation_text="50% Time Pressure", annotation_position="bottom left")
    fig_times.add_hline(y=0.8, line_width=2, line_dash="dash", line_color="#d31510", annotation_text="80% Time Pressure", annotation_position="bottom left")
    fig_times.update_layout(
        autosize=False,
        width=800,
        height=260,
        title="R.A. Time Pressure Plot",
        xaxis=dict(title="R.A.", range=[360, 0]),  # reversed to match sky map RA direction
        yaxis=dict(title="Fraction of 1-year time", range=[0, 1]),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    st.plotly_chart(fig_times, use_container_width=True)

st.divider()
st.header("Step 3: Save to cloud")
st.markdown("""
The JSON file contents in the editor will be saved to a remote database.
            
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
    mongo_uri = st.secrets["MONGO_URI"]
    if not mongo_uri:
        return False, "MONGO_URI environment variable not set."

    db_name = st.secrets["MONGO_DB"]
    coll_name = st.secrets["MONGO_COLLECTION"]

    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # trigger connection check
        client.server_info()

        db = client[db_name]
        coll = db[coll_name]

        doc = {
            "filename": filename,
            "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
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
# Inject CSS: disabled buttons grey, enabled buttons green/strong
st.markdown(
    """
    <style>
    /* Disabled buttons (greyed out) */
    .stButton>button[disabled] {
        background-color: #6c757d !important; /* grey */
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 0.6 !important;
        border: none !important;
    }
    /* Enabled buttons (emphasised, green) */
    .stButton>button:not([disabled]) {
        background-color: #28a745 !important; /* green */
        color: #ffffff !important;
        font-weight: 800 !important;
        font-size: 1.05rem !important;
        padding: 0.7rem 1.2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 6px 18px rgba(40,167,69,0.25) !important;
        border: none !important;
    }
    .stButton>button:not([disabled]):hover {
        background-color: #218838 !important;
    }
    .stButton {
        display: inline-block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.download_button(
    label="Download JSON File (optional)",
    data=json_string,
    file_name=fileOutputName,
    mime="application/json",
)

st.markdown("### Save to remote database", unsafe_allow_html=True)
# password input
pw = st.text_input("Upload password (press enter to confirm)", type="password", key="upload_pw_input", help="Enter password to enable Save")
# expected password from secrets
expected_pw = None
try:
    expected_pw = st.secrets.get("UPLOAD_PASSWORD")
except Exception:
    expected_pw = None

if expected_pw is None:
    st.warning("Upload password not configured in secrets. Saving is disabled.")
    save_disabled = True
else:
    save_disabled = (pw != expected_pw)

if pw and expected_pw and pw != expected_pw:
    st.error("Password incorrect.")

# Save button disabled until correct password entered
if st.button("Save to remote DB", key="save_remote_db", disabled=save_disabled):
    ok, msg = save_to_remote_db(json_string, fileOutputName)
    if ok:
        st.success(msg)
    else:
        st.error(msg)

st.divider()
st.header("Latest submissions")
if not latest_submissions:
    st.info("No submissions found in the remote DB.")
else:
    rows = []
    for s in sorted(latest_submissions.keys()):
        ent = latest_submissions[s]
        data_field = ent.get("data", {}) or {}
        justification = data_field.get("scienceJustification", "")
        n_areas = len(data_field.get("year1Areas", [])) if isinstance(data_field, dict) else ""
        rows.append({
            "survey": s,
            "timestamp": ent.get("timestamp"),
            "filename": ent.get("filename"),
            "n_areas": n_areas,
            "justification_preview": (justification[:120] + "…") if justification and len(justification) > 120 else justification
        })

    # provide expanders to inspect full stored document per submission
    for r in rows:
        with st.expander(f"{r['survey']} — {r.get('timestamp','')}", expanded=False):
            doc = latest_submissions.get(r['survey'])
            st.json(doc)

    allSubmissions = 'LTSYear1_all_surveys'+'_'+str(today.year)+today.strftime('%m')+today.strftime('%d')+'.json'
    
    # make this specific download button appear green like the Save button
    st.markdown(
        """
        <style>
        .green-download-btn .stButton>button[disabled] {
            background-color: #6c757d !important; /* grey */
            color: #ffffff !important;
            font-weight: 600 !important;
            opacity: 0.6 !important;
            border: none !important;
        }
        .green-download-btn .stButton>button:not([disabled]) {
            background-color: #28a745 !important; /* green */
            color: #ffffff !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            padding: 0.7rem 1.2rem !important;
            border-radius: 8px !important;
            box-shadow: 0 6px 18px rgba(40,167,69,0.25) !important;
            border: none !important;
        }
        .green-download-btn .stButton>button:not([disabled]):hover {
            background-color: #218838 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="green-download-btn">', unsafe_allow_html=True)
    st.download_button(
        label="Download all survey submissions",
        data=latest_submissions_json,
        file_name=allSubmissions,
        mime="application/json",
        key="download_all_submissions",
    )
    st.markdown('</div>', unsafe_allow_html=True)
st.divider()
st.header("Example JSON inputs")
st.markdown("""Here are three example polygons you can copy, paste, and edit!

All units are in degrees.
""")

c1, c2, c3 = st.columns((1, 1, 1))
c1.header("Decl. Stripe")
c1.json(      {
        "name": "Demo Stripe",
        "type": "stripe",
        "RA_lower": 0.0,
        "RA_upper": 52.5,
        "Dec_lower":-35.0,
        "Dec_upper":-25.0,
        "t_frac": 0.2
      })
c2.header('Single 4MOST Pointing')
c2.json({
    "name": "single_point",
    "type": "point",
    "RA_center":150.125,
    "Dec_center":2.2,
    "t_frac": 0.9
      })