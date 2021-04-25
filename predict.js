require("dotenv").config();

const util = require("util");
const fs = require("fs");
const sharp = require("sharp");
const msRest = require("@azure/ms-rest-js");
const backOff = require("exponential-backoff").backOff;
const _ = require("underscore");

const TrainingApi = require("@azure/cognitiveservices-customvision-training");
const PredictionApi = require("@azure/cognitiveservices-customvision-prediction");

const trainingKey = process.env.TRAINING_KEY;
const predictionKey = process.env.PREDICTION_KEY;
const predictionResourceId = process.env.PREDICTION_RESOURCE_ID;
const endPoint = process.env.ENDPOINT;

const publishIterationName = "classifyModel";
const setTimeoutPromise = util.promisify(setTimeout);

const credentials = new msRest.ApiKeyCredentials({
  inHeader: { "Training-key": trainingKey },
});
const trainer = new TrainingApi.TrainingAPIClient(credentials, endPoint);
const predictor_credentials = new msRest.ApiKeyCredentials({
  inHeader: { "Prediction-key": predictionKey },
});
const predictor = new PredictionApi.PredictionAPIClient(
  predictor_credentials,
  endPoint
);

function sharedBox(...boxes) {
  let sharedBox = boxes[0];
  for (const box of boxes) {
    sharedBox.left = Math.min(sharedBox.left, box.left);
    sharedBox.left2 = Math.max(sharedBox.left2, box.left2);
    sharedBox.top = Math.min(sharedBox.top, box.top);
    sharedBox.top2 = Math.max(sharedBox.top2, box.top2);
  }
  return sharedBox;
}

function textIntersects(b1, b2) {
  if (b1.left > b2.left) {
    const tmp = b2;
    b2 = b1;
    b1 = tmp;
  }

  return (
    b1.left < b2.left &&
    b1.left2 > b2.left &&
    b1.top < b2.top &&
    b1.top2 > b2.top
  );
}

function combineBoundingBoxes(predictions) {
  const boxes = [];
  const predictedBoxes = predictions
    .map((pred) => pred?.boundingBox || pred)
    .map((box) =>
      box?.left2
        ? box
        : {
            ...box,
            left2: box.left + box.width,
            top2: box.top + box.height,
            width: undefined,
            height: undefined,
          }
    );
  for (prediction of predictedBoxes) {
    const intersects = boxes.filter((existingBox) =>
      textIntersects(existingBox, prediction)
    );
    if (intersects.length > 0) {
      const newBox = sharedBox(...[...intersects, prediction]);
      for (oldIntersect of intersects) {
        intersects.splice(intersects.indexOf(oldIntersect), 1);
      }
      intersects.push(newBox);
    } else {
      boxes.push(prediction);
    }
  }
  return predictedBoxes;
}

async function predictCroppedImage(file) {
  const threshold = 0.6;

  const results = await backOff(() =>
    predictor.detectImage(process.env.PROJECT_ID, "Iteration3", file)
  );

  const predictions = results.predictions.filter(
    (prediction) => prediction.probability > threshold
  );

  return combineBoundingBoxes(predictions);
}

async function predictFullImage(path) {
  const boxSizes = 4000;
  const overflow = 100;

  const image = sharp(path);
  const meta = await image.metadata();

  const rows = Math.ceil(meta.height / (boxSizes - overflow));
  const cols = Math.ceil(meta.width / (boxSizes - overflow));

  const predictions = [];

  for (let row = 0; row < rows - 1; row++) {
    for (let col = 0; col < cols - 1; col++) {
      const predictor = async () => {
        const imageSubset = await image
          .extract({
            left: col * (boxSizes - overflow),
            top: row * (boxSizes - overflow),
            width: boxSizes,
            height: boxSizes,
          })
          .toBuffer();
        return [
          await predictCroppedImage(imageSubset),
          col * (boxSizes - overflow),
          row * (boxSizes - overflow),
        ];
      };
      predictions.push(predictor());
    }
  }

  const results = await Promise.all(predictions);

  //We need to calculate bounding boxes in our coordinate space
  const coordinateMappedResults = results.map((picResult) => {
    const boxes = picResult[0];
    const leftStart = picResult[1];
    const topStart = picResult[2];

    return boxes.map((box) => ({
      left: leftStart + box.left * boxSizes,
      top: topStart + box.top * boxSizes,
      left2: leftStart + box.left2 * boxSizes,
      top2: topStart + box.top2 * boxSizes,
    }));
  });

  const flatBoxes = coordinateMappedResults.flat();
  const result = combineBoundingBoxes(flatBoxes);
  console.log(result);

  return combineBoundingBoxes(flatBoxes);
}

predictFullImage("./assets/F_7.png");
