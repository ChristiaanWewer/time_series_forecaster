# Deep Time Series Forecaster

A PyTorch library for deep learning on time series that I am building out of my own interest, so I can implement papers and ideas I find interesting and test them on real data. My specific interests are in **embeddings**, **probabilistic forecasting** and **long range forecasting**, and the library is designed so that each of these can be added as a small, self-contained component instead of a rewrite.

> **Use at your own risk.** This is a personal hobby research project and the API still changes dramatically. I am currently implementing a CUDA prefetcher and efficient loading of large time series (a disk-backed data format)..

## Implemented papers

### DILATE: shape and time distortion loss

[Le Guen & Thome, *Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models*, NeurIPS 2019](https://papers.nips.cc/paper_files/paper/2019/hash/466accbac9a66b805ba50e42ad715740-Abstract.html)

A model trained with MSE or MAE is rewarded for predicting the average of all plausible futures, and with that it smooths out sharp changes. For many applications, such as peak discharges in a river, the peak is the most important part of the forecast. DILATE splits the error into two parts: a *shape* term based on soft dynamic time warping, which rewards getting the form of the signal right, such as the height and steepness of a peak, and a *temporal* term, which penalises predicting that peak too early or too late. The result is a forecast that better captures peaks and sudden changes instead of a flattened curve.

My implementation is a batched Numba port of the reference implementation, which is fast enough to train on without `torch.compile`, and in which both terms contribute to the gradient. The results are verified against the upstream package.

### Regularization Learning Networks (RLN)

[Shavitt & Segal, *Regularization Learning Networks*, NeurIPS 2018](https://arxiv.org/abs/1805.06440)

Ordinary L1 or L2 regularization uses one coefficient for every weight in the network. RLN learns a separate L1 coefficient for *each individual weight*, jointly with the weights themselves, by minimising a counterfactual loss: how the loss on the next batch would have changed with a different regularization strength. Because the coefficients are pinned to a fixed mean, regularization is redistributed between weights rather than vanishing. In practice this creates a sparse network in which uninformative inputs and connections are pushed to zero, which helps on data where only a few inputs matter, reduces overfitting on small datasets and makes the resulting network easier to interpret.

RLN can be applied to the embedding networks, to the model backbone or to both, independently of each other.

## What I am working on and towards

### Learning across many time series and timescales

Much of my thinking comes from hydrology, where a single model is trained on hundreds of catchments at once, and where [neuralhydrology](https://github.com/neuralhydrology/neuralhydrology) showed how well that works. I adopted several ideas from that codebase:

- **Exact normalization statistics with Chan's parallel algorithm.** Means and variances are merged source by source, so the statistics are exact while never more than one series needs to be in memory. The same algorithm gives exact *pooled* validation metrics over a whole validation set, rather than an average of per-batch values, which is wrong for ratio-based metrics such as NSE.
- **Rescaling the loss per series.** When series of very different magnitude are trained together, the series with the largest values dominates the gradients. Following neuralhydrology's NSE\* loss, each sample's error is divided by the standard deviation (for MAE and quantile loss) or variance (for MSE and expectile loss) of its own series, so that every series contributes equally while batches stay randomly mixed.
- **Hydrological metrics** such as NSE and its α and β decomposition, verified against neuralhydrology's implementation.

The next step is multi-timescale learning: one model that predicts at for example hourly and daily resolution, in the spirit of the MTS-LSTM work from the same community.

### Beyond hydrology

Hydrology is where I started, but I do not want the library to be limited to it. I also want to apply it to other domains, such as **demand forecasting** on the [M5 dataset](https://www.kaggle.com/competitions/m5-forecasting-accuracy) and **electricity demand**, and to other datasets that come along. These problems share the same core challenge of learning across many related series of very different magnitude, but each has its own statistical properties, and working with many different types of datasets lets me study how methods behave under each of them. M5, for example, has intermittent sales with many zeros and a hierarchy of products and stores. Electricity demand has strong daily and weekly seasonality and depends heavily on weather and calendar effects. River discharge, in turn, is dominated by rare, sharp peaks. Testing on several domains also keeps the library general instead of tuned to one type of data.

### Probabilistic forecasting

A single number hides how certain a forecast is, and for decisions such as flood warnings the uncertainty is as important as the forecast itself. Right now the library has quantile regression with **non-crossing quantiles**: the output head predicts the median directly and builds every other quantile as a cumulative positive offset from it, so a 90% quantile can never fall below an 80% quantile, by construction. Expectile regression is implemented in the same way. Forecasts are evaluated with CRPS, a scale-invariant CRPS for comparing series of different magnitude, the Winkler interval score, coverage and interval width.

Quantile regression does not guarantee that a 90% interval actually covers 90% of the observations. I plan to add **conformal prediction** to obtain calibrated intervals that do. Conformal prediction gives this coverage guarantee under the assumption that the data is *exchangeable*, meaning the order of the observations does not matter. Time series break this assumption by definition, through autocorrelation, seasonality and distribution shift, so I will look into variants that relax it, such as adaptive conformal inference that keeps adjusting its intervals as new errors come in.

### Embeddings

How a raw input is represented before it enters the model matters more than it is often given credit for. The library currently supports learned embeddings per group of input variables, optionally shared between the historic and future inputs. Three directions I want to develop:

- **Piecewise linear encoding.** [Gorishniy et al. (2022)](https://arxiv.org/pdf/2203.05556) showed that encoding a numerical feature as its position within a set of bins, instead of a single scalar, substantially improves deep models on tabular data. The bins are fixed beforehand from quantiles or a decision tree, and I plan to develop a learned variant in which the bin edges are trained together with the model.
- **Merging different data products.** In environmental modelling the same quantity, such as precipitation, is often available from several data products, each with its own gaps and biases. [Masked mean embeddings](https://hess.copernicus.org/articles/29/6221/2025/) embed each product separately and average the embeddings of the products that are available at each time step, so the model handles missing products naturally instead of discarding those time steps.
- **Multi-timescale embeddings.** Embeddings that naturally incorporate information at several timescales at once. I have ongoing ideas for this that I will implement later.

## Quick start

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and the project requires Python 3.14.

```bash
uv sync
uv run pytest
```

Every component is a plain Python object that you construct directly, there are no config dicts or string specifications. A minimal probabilistic forecast looks like this:

```python
from src.forecaster import Forecaster
from src.models import LSTMHistoric
from src.data import DataSource, TimeSeriesData
from src.optim import Adam
from src.utils.scores_and_losses import QuantileLoss, CRPS, CoverageGap

def source(start, end):
    return DataSource(csv='data/data.csv', start=start, end=end, csv_index_col=1, nodata_values=[-999])

train = TimeSeriesData([source('1980-01-01', '2004-12-31')])
val = TimeSeriesData([source('2005-01-01', '2009-12-31')])

fc = Forecaster(
    model=LSTMHistoric(hidden_size=28, num_layers=1, dropout_rate=0.0),
    name='lstm_quantile',
    loss=QuantileLoss(q=[0.05, 0.5, 0.95]),
    validation_score=CRPS(),
    validation_logging_criteria=[CoverageGap(alpha=0.1)],
    optimizer=Adam(lr=0.001),
    training_data=train,
    validation_data=val,
    historic_cols=['prcp', 'tmax', 'tmin', 'vp', 'srad'],
    target_col='target',
    forecasting_horizon=7,
    historic_input_sequence_length=30,
    number_of_epochs=30,
    save_path='models',
)
fc.fit()
```

The notebooks in [examples/](examples/) show complete runs:

| Notebook | Shows |
|---|---|
| [example_model_and_loss_comparison.ipynb](examples/example_model_and_loss_comparison.ipynb) | LSTM architectures trained with MAE versus DILATE |
| [example_regularization_rln.ipynb](examples/example_regularization_rln.ipynb) | RLN versus no regularization |
| [example_probabilistic_forecasting.ipynb](examples/example_probabilistic_forecasting.ipynb) | Quantile and expectile regression with probabilistic scores |
| [example_embeddings.ipynb](examples/example_embeddings.ipynb) | Embedding networks, including one shared between historic and future inputs |

## Structure

```
src/
  data/         DataSource, TimeSeriesData (in memory or on disk), windowing, normalization
  forecaster/   Forecaster: training loop, validation, checkpointing, prediction; CUDA prefetcher
  models/       LSTM backbones, embedding networks, output heads (incl. non-crossing quantiles)
  optim.py      optimizer wrappers
  utils/        losses and scores, regularizers (L1, L2, RLN), DILATE
tests/          pytest suite
examples/       notebooks
```
