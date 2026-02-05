import streamlit as st
import yfinance as yf
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="QInvestor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 QInvestor")
st.caption("Performance, peer comparison, rolling trends, correlations & Quantum Predictions")

# ---------------- INDEX CONFIG ----------------
INDICES = {
    "Sensex": "^BSESN",
    "Nifty 50": "^NSEI",
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
    "Switzerland 20": "^SSMI", 
    "MDAX": "^MDAXI", 
    "BEL 20": "^BFX", 
    "DAX": "^GDAXI",
    "CAC 40": "^FCHI", 
    "AEX": "^AEX", 
    "EURO STOXX 50": "^STOXX50E", 
    "IBEX 35": "^IBEX", 
    "TECDAX": "^TECDAX", 
    "OMX Stockholm 30": "^OMX", 
    "OMX Helsinki 25": "^OMXH25", 
    "S&P 100": "^SP100"
}

DEFAULT_INDICES = []

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Controls")

selected_indices = st.sidebar.multiselect(
    "Select Indices",
    list(INDICES.keys()),
    default=DEFAULT_INDICES
)

if not selected_indices:
    st.stop()

# -------- TIME PERIOD --------
horizon_map = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y"
}

period = st.sidebar.selectbox(
    "Time Period",
    list(horizon_map.keys()),
    index=2
)

# -------- ROLLING WINDOW CONTROL --------
rolling_window = st.sidebar.slider(
    "Rolling Average Window (days)",
    min_value=5,
    max_value=100,
    value=20
)

# ---------------- DATA LOADER ----------------
@st.cache_data(ttl=3600)
def load_data(indices, period):
    tickers = [INDICES[i] for i in indices]
    return yf.download(
        tickers,
        period=period,
        group_by="ticker",
        progress=False
    )

raw_data = load_data(selected_indices, horizon_map[period])

# ---------------- CLOSE PRICES ----------------
close_df = pd.DataFrame()
for idx in selected_indices:
    tkr = INDICES[idx]
    if tkr in raw_data:
        close_df[idx] = raw_data[tkr]["Close"]
close_df.dropna(inplace=True)
if close_df.empty:
    st.error("No data available")
    st.stop()

# ---------------- NORMALIZATION ----------------
normalized = close_df.div(close_df.iloc[0])
latest = normalized.iloc[-1]

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Performance",
    "📊 Peer Comparison",
    "📉 Rolling Averages",
    "🔗 Correlation",
    "🕯️ Candlesticks",
    "🔮 Quantum Prediction"
])

# ---------------- TAB 1: PERFORMANCE ----------------
with tab1:
    c1, c2 = st.columns(2)
    c1.metric("📈 Best Performer", latest.idxmax(), f"{(latest.max()-1)*100:.2f}%")
    c2.metric("📉 Worst Performer", latest.idxmin(), f"{(latest.min()-1)*100:.2f}%")
    
    st.subheader("📈 Normalized Performance")
    norm_df = normalized.reset_index().melt(
        id_vars="Date",
        var_name="Index",
        value_name="Normalized Value"
    )
    st.altair_chart(
        alt.Chart(norm_df).mark_line().encode(
            x="Date:T",
            y="Normalized Value:Q",
            color="Index:N",
            tooltip=["Date", "Index", "Normalized Value"]
        ).properties(height=400).interactive(),
        use_container_width=True
    )

# ---------------- TAB 2: PEER COMPARISON ----------------
with tab2:
    if len(selected_indices) > 1:
        st.subheader("📊 Index vs Peer Average")
        cols = st.columns(2)
        for i, idx in enumerate(selected_indices):
            peers = normalized.drop(columns=idx)
            peer_avg = peers.mean(axis=1)
            df = pd.DataFrame({
                "Date": normalized.index,
                idx: normalized[idx],
                "Peer Average": peer_avg
            }).melt("Date", var_name="Series", value_name="Value")
            chart = alt.Chart(df).mark_line().encode(
                x="Date:T",
                y="Value:Q",
                color="Series:N",
                strokeDash=alt.condition(
                    alt.datum.Series == "Peer Average",
                    alt.value([5, 5]),
                    alt.value([0])
                )
            ).properties(title=f"{idx} vs Peer Average", height=300)
            cols[i % 2].altair_chart(chart, use_container_width=True)
    else:
        st.info("Select at least 2 indices for peer comparison")

# ---------------- TAB 3: ROLLING AVERAGES ----------------
with tab3:
    st.subheader("📉 Rolling Averages")
    rolling_df = close_df.rolling(window=rolling_window).mean()
    roll_df = rolling_df.reset_index().melt(
        id_vars="Date",
        var_name="Index",
        value_name="Rolling Avg"
    )
    st.altair_chart(
        alt.Chart(roll_df).mark_line().encode(
            x="Date:T",
            y="Rolling Avg:Q",
            color="Index:N",
            tooltip=["Date", "Index", "Rolling Avg"]
        ).properties(height=350),
        use_container_width=True
    )

# ---------------- TAB 4: CORRELATION ----------------
with tab4:
    st.subheader("🔗 Correlation Matrix")
    returns = close_df.pct_change().dropna()
    if len(returns) > 5 and len(selected_indices) > 1:
        corr = returns.corr()
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for correlation analysis")

# ---------------- TAB 5: CANDLESTICKS ----------------
with tab5:
    st.subheader("🕯️ Candlestick Charts")
    st.markdown("### 💾 Download Full OHLC Data")
    full_ohlc = pd.concat([
        raw_data[INDICES[idx]][["Open", "High", "Low", "Close"]].assign(Index=idx)
        for idx in selected_indices
    ])
    full_ohlc = full_ohlc.reset_index()
    st.dataframe(full_ohlc)
    csv_full_ohlc = full_ohlc.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Full OHLC Data CSV",
        data=csv_full_ohlc,
        file_name="full_ohlc_data.csv",
        mime="text/csv"
    )
    cols = st.columns(2)
    for i, idx in enumerate(selected_indices):
        tkr = INDICES[idx]
        try:
            df = raw_data[tkr].dropna()
            fig = go.Figure(go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"]
            ))
            fig.update_layout(
                title=f"{idx} Candlestick",
                xaxis_rangeslider_visible=False,
                height=350
            )
            cols[i % 2].plotly_chart(fig, use_container_width=True)
        except:
            st.warning(f"Could not load {idx}")

# ---------------- TAB 6: QUANTUM PREDICTION ----------------
with tab6:
    st.header("🔮 Quantum Neural Network Stock Prediction")
    
    # Inputs
    qnn_ticker = st.text_input("Enter Stock Ticker for Quantum Prediction:", "AAPL", key="qnn_ticker")
    qnn_start = st.date_input("Start Date", pd.to_datetime("2023-01-01"), key="qnn_start")
    qnn_end = st.date_input("End Date", pd.to_datetime("2023-12-31"), key="qnn_end")
    
    if st.button("Run Quantum Prediction", key="run_qnn"):
        import matplotlib.pyplot as plt
        import pennylane as qml
        from pennylane import numpy as pnp
        import jax
        from jax import numpy as jnp
        import optax
        from sklearn.metrics import r2_score,mean_squared_error
        import numpy as np
        import yfinance as yf
        import pandas as pd
        import datetime
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        import warnings

        warnings.filterwarnings("ignore")
        pnp.random.seed(42)

        dev1 = qml.device('default.qubit', wires=2)

        # Load stock data
        data = yf.download(qnn_ticker, start=qnn_start, end=qnn_end)
        if data.empty:
            st.warning("No data found for this ticker and date range!")
            st.stop()

        data['target'] = data['Close'].shift(-1).fillna(method='ffill')

        X=data[['Close']].to_numpy()
        y=data[['target']].to_numpy()

        scaler=StandardScaler() # MinMaxScaler()
        X=scaler.fit_transform(X)
        y=scaler.fit_transform(y)

        # ---------------- QUANTUM RESERVOIR ----------------
        dev2=qml.device('lightning.qubit',wires=range(1))

        @qml.qnode(device=dev2)
        def quantum_reservoir(data_point):
            for value in data_point:
                qml.Hadamard(wires=0)
                qml.RX(2*pnp.pi*value,wires=0)
                qml.RY(pnp.pi*value,wires=0)
                qml.RZ(3*pnp.pi*value,wires=0)
            return qml.probs(wires=range(1))

        def extract_features(data):
            return [quantum_reservoir(d) for d in data]

        @qml.qnode(device=dev2)
        def target_encoder(data_point):
            for value in data_point:
                qml.Hadamard(wires=0)
                qml.RX(2*pnp.pi*value,wires=0)
                qml.RY(pnp.pi*value,wires=0)
                qml.RZ(3*pnp.pi*value,wires=0)
            return qml.expval(qml.PauliZ(0))

        def extract_target(data):
            return [target_encoder(d) for d in data]

        # ---------------- QUANTUM NEURAL NETWORK ----------------
        dev1 = qml.device('default.qubit', wires=2)

        # Quantum Neural Network

        def S(x):
            qml.AngleEmbedding( x, wires=[0,1],rotation='Z')

        def W(params):
            qml.StronglyEntanglingLayers(params, wires=[0,1])

        @qml.qnode(dev1,interface="jax")
        def quantum_neural_network(params, x):
            layers=len(params[:,0,0])-1
            n_wires=len(params[0,:,0])
            n_params_rot=len(params[0,0,:])
            for i in range(layers):
                W(params[i,:,:].reshape(1,n_wires,n_params_rot))
                S(x)
            W(params[-1,:,:].reshape(1,n_wires,n_params_rot))

            return qml.expval(qml.PauliZ(wires=0)@qml.PauliZ(wires=1))

        # Optimization for the quantum neural network algorithm

        @jax.jit
        def mse(params,x,targets):
            # We compute the mean square error between the target function and the quantum circuit to quantify the quality of our estimator
            return (quantum_neural_network(params,x)-jnp.array(targets))**2
        @jax.jit
        def loss_fn(params, x,targets):
            # We define the loss function to feed our optimizer
            mse_pred = jax.vmap(mse,in_axes=(None, 0,0))(params,x,targets)
            loss = jnp.mean(mse_pred)
            return loss

        opt = optax.adam(learning_rate=0.05)
        max_steps=300

        @jax.jit
        def update_step_jit(i, args):
            # We loop over this function to optimize the trainable parameters
            params, opt_state, data, targets, print_training = args
            loss_val, grads = jax.value_and_grad(loss_fn)(params, data, targets)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            def print_fn():
                jax.debug.print("Step: {i}  Loss: {loss_val}", i=i, loss_val=loss_val)
            # if print_training=True, print the loss every 50 steps
            jax.lax.cond((jnp.mod(i, 50) == 0 ) & print_training, print_fn, lambda: None)
            return (params, opt_state, data, targets, print_training)

        @jax.jit
        def optimization_jit(params, data, targets, print_training=False):
            opt_state = opt.init(params)
            args = (params, opt_state, jnp.asarray(data), targets, print_training)
            # We loop over update_step_jit max_steps iterations to optimize the parameters
            (params, opt_state, _, _, _) = jax.lax.fori_loop(0, max_steps+1, update_step_jit, args)
            return params

        def evaluate(params, data):
            y_pred = jax.vmap(quantum_neural_network, in_axes=(None, 0))(params, data)
            return y_pred

        # ---------------- EVALUATE ----------------

        X=extract_features(X)
        y=extract_target(y)
        X=pnp.array(X)
        y=pnp.array(y)

        # Running the quantum neural network
        wires=2
        layers=4
        params_shape = qml.StronglyEntanglingLayers.shape(n_layers=layers+1,n_wires=wires)
        params=pnp.random.default_rng().random(size=params_shape)
        best_params=optimization_jit(params, X, jnp.array(y), print_training=True)

        y_predictions=evaluate(best_params,X)

        st.subheader("Predictions vs Actuals")
        st.write(pd.DataFrame({
            "Predicted": y_predictions[0:5],
            "Actual": y[0:5]
        }))
        st.write(f"Mean Squared Error: {mean_squared_error(y,y_predictions):.4f}")
        st.write(f"R² Score: {r2_score(y,y_predictions):.4f}")

        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(y[0:99], label="Actual", linewidth=2)
        ax.plot(y_predictions[0:99], label="Predicted", linestyle="--")
        ax.legend()
        ax.set_title("QNN Based Stock Price Prediction (Quantum Data)")

        st.pyplot(fig)



