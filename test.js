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

const generatePermutations = (a1, a2) => {
  const result = [];

  for (const i1 of a1) {
    for (const i2 of a2) {
      result.push([i1, i2]);
    }
  }
  return result;
};

const generateTestPermutations = async (fileName, tag, projectId) => {
  const image = sharp(`TrainingPictograms/${fileName}`);
  const imageMetadata = await image.metadata();

  const rotatePermutations = [10, 55, 45, 90, 111, 135, 180, 225, 270, 315];
  const resizePermutations = [1, 1.2, 0.9, [1, 1.2], [1.3, 1]];

  const permutationConfig = generatePermutations(
    rotatePermutations,
    resizePermutations
  );

  const permutations = permutationConfig.map(([rotation, scale]) => {
    if (!Array.isArray(scale)) {
      return (image) => {
        return [
          tag,
          image
            .rotate(rotation, { background: "#fff" })
            .resize(
              Math.round(imageMetadata.width * scale),
              Math.round(imageMetadata.height * scale)
            ),
        ];
      };
    } else {
      return (image) => {
        return [
          tag,
          image
            .rotate(rotation, { background: "#fff" })
            .resize(
              Math.round(imageMetadata.width * scale[0]),
              Math.round(imageMetadata.height * scale[1])
            ),
        ];
      };
    }
  });

  const testImages = await Promise.all(
    permutations.map(async (permutation) => {
      const newImage = sharp(await image.png().toBuffer());
      const transformed = permutation(newImage);
      return transformed;
    })
  );

  return testImages;
};

const createTagIfNotExisting = async (projectId, tagName) => {
  const tags = await trainer.getTags(projectId);
  return (
    tags.find((tag) => tag.name === tagName) ||
    trainer.createTag(projectId, tagName)
  );
};

const testTrain = async () => {
  const sampleProject = await trainer.getProject(process.env.PROJECT_ID);
  const fileUploadPromises = [];

  const files = await fs.promises.readdir("TrainingPictograms");
  const tags = await Promise.all(
    files.map(async (file) => {
      return [
        file,
        await createTagIfNotExisting(
          sampleProject.id,
          file.replace(".png", "")
        ),
      ];
    })
  );
  const permutationIcons = _.union(
    ...(await Promise.all(
      tags.map((tag) =>
        generateTestPermutations(tag[0], tag[1], sampleProject.id)
      )
    ))
  );

  for (let i = 0; i < 50; i++) {
    const regions = [];
    let trainingImage = sharp({
      create: {
        width: 1000,
        height: 1000,
        channels: 3,
        background: { r: 255, g: 255, b: 255 },
      },
    });
    for (let j = 0; j < 20; j++) {
      const randomPermutation =
        permutationIcons[_.random(0, permutationIcons.length - 1)];
      const testImageMeta = await sharp(
        await randomPermutation[1].png().toBuffer()
      ).metadata();
      const randomLeft = _.random(0, 800);
      const randomTop = _.random(0, 800);
      regions.push({
        tagId: randomPermutation[0].id,
        left: (randomLeft - 5) / 1000,
        top: (randomTop - 5) / 1000,
        width: (testImageMeta.width + 5) / 1000,
        height: (testImageMeta.height + 5) / 1000,
      });

      trainingImage = sharp(
        await trainingImage
          .composite([
            {
              input: await randomPermutation[1].png().toBuffer(),
              left: randomLeft,
              top: randomTop,
            },
          ])
          .png()
          .toBuffer()
      );
    }
    await backOff(async () => {
      const batch = {
        images: [
          {
            contents: await trainingImage.png().toBuffer(),
            regions,
          },
        ],
      };
      const result = await trainer.createImagesFromFiles(
        sampleProject.id,
        batch
      );
      console.log(result);
    });
  }
};

testTrain();
