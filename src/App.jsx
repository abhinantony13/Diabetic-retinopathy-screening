import { useEffect, useState } from "react";
import "./App.css";

/* =========================================================
   RETINEX
   Explainable AI Retinal Screening
   ========================================================= */

const API_URL = "http://127.0.0.1:8000";

const DR_LEVELS = {
  0: {
    name: "No Diabetic Retinopathy",
    short: "No DR",
    colorClass: "level-zero",
    description:
      "No diabetic retinopathy was identified by the pretrained classifier.",
  },

  1: {
    name: "Mild Non-Proliferative Diabetic Retinopathy",
    short: "Mild NPDR",
    colorClass: "level-one",
    description:
      "The classifier predicts mild non-proliferative diabetic retinopathy.",
  },

  2: {
    name: "Moderate Non-Proliferative Diabetic Retinopathy",
    short: "Moderate NPDR",
    colorClass: "level-two",
    description:
      "The classifier predicts moderate non-proliferative diabetic retinopathy.",
  },

  3: {
    name: "Severe Non-Proliferative Diabetic Retinopathy",
    short: "Severe NPDR",
    colorClass: "level-three",
    description:
      "The classifier predicts severe non-proliferative diabetic retinopathy.",
  },

  4: {
    name: "Proliferative Diabetic Retinopathy",
    short: "PDR",
    colorClass: "level-four",
    description:
      "The classifier predicts proliferative diabetic retinopathy.",
  },
};


/* =========================================================
   IMAGE QUALITY ANALYSIS
   ========================================================= */

function calculateImageQuality(imageElement) {
  const canvas = document.createElement("canvas");

  const ctx = canvas.getContext("2d", {
    willReadFrequently: true,
  });

  const MAX_SIZE = 400;

  const scale = Math.min(
    1,
    MAX_SIZE /
      Math.max(
        imageElement.naturalWidth,
        imageElement.naturalHeight
      )
  );

  const width = Math.max(
    1,
    Math.round(
      imageElement.naturalWidth * scale
    )
  );

  const height = Math.max(
    1,
    Math.round(
      imageElement.naturalHeight * scale
    )
  );

  canvas.width = width;
  canvas.height = height;

  ctx.drawImage(
    imageElement,
    0,
    0,
    width,
    height
  );

  const imageData = ctx.getImageData(
    0,
    0,
    width,
    height
  );

  const data = imageData.data;
  const totalPixels = width * height;

  let brightnessSum = 0;
  let brightnessSquared = 0;

  let darkPixels = 0;
  let brightPixels = 0;

  const gray = new Float32Array(
    totalPixels
  );


  /* -------------------------------------------------------
     Grayscale
     ------------------------------------------------------- */

  for (
    let i = 0, p = 0;
    i < data.length;
    i += 4, p++
  ) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];

    const value =
      0.299 * r +
      0.587 * g +
      0.114 * b;

    gray[p] = value;

    brightnessSum += value;
    brightnessSquared +=
      value * value;

    if (value < 20) {
      darkPixels++;
    }

    if (value > 245) {
      brightPixels++;
    }
  }


  const meanBrightness =
    brightnessSum /
    totalPixels;

  const variance =
    brightnessSquared /
      totalPixels -
    meanBrightness *
      meanBrightness;

  const brightnessStd =
    Math.sqrt(
      Math.max(variance, 0)
    );


  /* -------------------------------------------------------
     Focus / Sharpness
     ------------------------------------------------------- */

  let laplacianSum = 0;
  let laplacianSquared = 0;
  let count = 0;

  for (
    let y = 1;
    y < height - 1;
    y++
  ) {
    for (
      let x = 1;
      x < width - 1;
      x++
    ) {
      const center =
        gray[y * width + x];

      const top =
        gray[(y - 1) * width + x];

      const bottom =
        gray[(y + 1) * width + x];

      const left =
        gray[y * width + x - 1];

      const right =
        gray[y * width + x + 1];

      const laplacian =
        top +
        bottom +
        left +
        right -
        4 * center;

      laplacianSum +=
        laplacian;

      laplacianSquared +=
        laplacian *
        laplacian;

      count++;
    }
  }

  const laplacianMean =
    laplacianSum /
    Math.max(count, 1);

  const laplacianVariance =
    laplacianSquared /
      Math.max(count, 1) -
    laplacianMean *
      laplacianMean;

  const focus = Math.round(
    Math.min(
      100,
      Math.max(
        0,
        100 *
          (
            1 -
            Math.exp(
              -laplacianVariance /
                150
            )
          )
      )
    )
  );


  /* -------------------------------------------------------
     Illumination
     ------------------------------------------------------- */

  const brightnessScore =
    Math.max(
      0,
      100 -
        Math.abs(
          meanBrightness - 120
        ) *
          0.7
    );

  const uniformityPenalty =
    Math.max(
      0,
      brightnessStd - 65
    );

  const clippedRatio =
    (
      darkPixels +
      brightPixels
    ) /
    totalPixels;

  const illumination =
    Math.round(
      Math.min(
        100,
        Math.max(
          0,
          brightnessScore -
            uniformityPenalty *
              0.5 -
            clippedRatio *
              30
        )
      )
    );


  /* -------------------------------------------------------
     Field of View
     ------------------------------------------------------- */

  const border = Math.max(
    2,
    Math.floor(
      Math.min(
        width,
        height
      ) * 0.05
    )
  );

  let borderPixels = 0;
  let visiblePixels = 0;

  for (
    let y = 0;
    y < height;
    y++
  ) {
    for (
      let x = 0;
      x < width;
      x++
    ) {
      const isBorder =
        x < border ||
        x >= width - border ||
        y < border ||
        y >= height - border;

      if (!isBorder) {
        continue;
      }

      borderPixels++;

      if (
        gray[
          y * width + x
        ] > 25
      ) {
        visiblePixels++;
      }
    }
  }

  const visibleRatio =
    visiblePixels /
    Math.max(
      borderPixels,
      1
    );

  const resolutionScore =
    Math.min(
      100,
      Math.min(
        imageElement.naturalWidth,
        imageElement.naturalHeight
      ) / 8
    );

  const fieldOfView =
    Math.round(
      Math.min(
        100,
        Math.max(
          0,
          60 +
            visibleRatio *
              25 +
            resolutionScore *
              0.15
        )
      )
    );


  /* -------------------------------------------------------
     Overall score
     ------------------------------------------------------- */

  const overall =
    Math.round(
      focus * 0.4 +
        illumination *
          0.35 +
        fieldOfView *
          0.25
    );


  let status;
  let feedback;

  if (
    overall >= 80 &&
    focus >= 60 &&
    illumination >= 60 &&
    fieldOfView >= 60
  ) {
    status = "GOOD";

    feedback =
      "Image quality is adequate for AI analysis.";
  } else if (
    overall >= 60
  ) {
    status = "BORDERLINE";

    const problems = [];

    if (focus < 60) {
      problems.push("focus");
    }

    if (illumination < 60) {
      problems.push(
        "illumination"
      );
    }

    if (fieldOfView < 60) {
      problems.push(
        "field of view"
      );
    }

    feedback =
      problems.length > 0
        ? `Borderline ${problems.join(
            ", "
          )}. Enhancement is recommended.`
        : "Image is borderline. Enhancement is recommended.";
  } else {
    status = "RECAPTURE";

    const problems = [];

    if (focus < 45) {
      problems.push(
        "improve focus"
      );
    }

    if (illumination < 45) {
      problems.push(
        "improve illumination"
      );
    }

    if (fieldOfView < 45) {
      problems.push(
        "capture more retinal field"
      );
    }

    feedback =
      problems.length > 0
        ? `Image is ungradeable. Please ${problems.join(
            " and "
          )} and recapture.`
        : "Image quality is insufficient. Please recapture.";
  }

  return {
    focus,
    illumination,
    fieldOfView,
    overall,
    status,
    feedback,
  };
}


/* =========================================================
   SMALL UI COMPONENTS
   ========================================================= */

function MetricBar({
  label,
  value,
}) {
  return (
    <div className="metric">

      <div className="metric-top">

        <span>
          {label}
        </span>

        <strong>
          {value !== null &&
          value !== undefined
            ? `${value}%`
            : "—"}
        </strong>

      </div>

      <div className="metric-track">

        <div
          className="metric-fill"
          style={{
            width:
              value !== null &&
              value !== undefined
                ? `${value}%`
                : "0%",
          }}
        />

      </div>

    </div>
  );
}


function StepLabel({
  number,
  children,
}) {
  return (
    <div className="step-label">

      <span>
        {number}
      </span>

      {children}

    </div>
  );
}


/* =========================================================
   MAIN APP
   ========================================================= */

function App() {
  const [image, setImage] =
    useState(null);

  const [imageFile, setImageFile] =
    useState(null);

  const [quality, setQuality] =
    useState(null);

  const [screening, setScreening] =
    useState(false);

  const [prediction, setPrediction] =
    useState(null);

  const [predictionError, setPredictionError] =
    useState("");

  const [activeNav, setActiveNav] =
    useState("Screening");


  /* -------------------------------------------------------
     Clean object URL
     ------------------------------------------------------- */

  useEffect(() => {
    return () => {
      if (image) {
        URL.revokeObjectURL(
          image
        );
      }
    };
  }, [image]);


  /* -------------------------------------------------------
     Upload image
     ------------------------------------------------------- */

  const handleImage = (
    event
  ) => {
    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    if (
      !file.type.startsWith(
        "image/"
      )
    ) {
      alert(
        "Please select a valid image file."
      );
      return;
    }

    if (
      file.size >
      10 * 1024 * 1024
    ) {
      alert(
        "Image must be smaller than 10 MB."
      );
      return;
    }

    if (image) {
      URL.revokeObjectURL(
        image
      );
    }

    const imageURL =
      URL.createObjectURL(
        file
      );

    const img =
      new Image();

    img.onload = () => {
      try {
        const result =
          calculateImageQuality(
            img
          );

        setImage(
          imageURL
        );

        setImageFile(
          file
        );

        setQuality(
          result
        );

        setPrediction(
          null
        );

        setPredictionError(
          ""
        );

      } catch (error) {
        console.error(
          error
        );

        URL.revokeObjectURL(
          imageURL
        );

        alert(
          "Unable to analyze this image."
        );
      }
    };

    img.onerror = () => {
      URL.revokeObjectURL(
        imageURL
      );

      alert(
        "Unable to read this image."
      );
    };

    img.src =
      imageURL;
  };


  /* -------------------------------------------------------
     Reset
     ------------------------------------------------------- */

  const changeImage =
    () => {
      if (image) {
        URL.revokeObjectURL(
          image
        );
      }

      setImage(null);
      setImageFile(null);
      setQuality(null);
      setPrediction(null);
      setPredictionError("");
      setScreening(false);
    };


  /* -------------------------------------------------------
     AI prediction
     ------------------------------------------------------- */

  const startScreening =
    async () => {

      if (
        !imageFile ||
        !quality
      ) {
        return;
      }

      if (
        quality.status ===
        "RECAPTURE"
      ) {
        return;
      }

      setScreening(
        true
      );

      setPrediction(
        null
      );

      setPredictionError(
        ""
      );

      try {

        const formData =
          new FormData();

        formData.append(
          "file",
          imageFile
        );

        const response =
          await fetch(
            `${API_URL}/predict`,
            {
              method:
                "POST",
              body:
                formData,
            }
          );

        const data =
          await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "AI prediction failed."
          );
        }

        setPrediction(
          data
        );

      } catch (error) {

        console.error(
          "Prediction error:",
          error
        );

        setPredictionError(
          error.message ||
            "Unable to connect to the RETINEX AI backend."
        );

      } finally {

        setScreening(
          false
        );
      }
    };


  const levelInfo =
    prediction
      ? DR_LEVELS[
          prediction.dr_level
        ]
      : null;


  return (
    <div className="retinex-app">

      {/* ===================================================
          SIDEBAR
          =================================================== */}

      <aside className="sidebar">

        <div className="brand">

          <div className="brand-mark">

            <div className="brand-eye">
              <span />
            </div>

          </div>

          <div className="brand-name">

            <strong>
              RETINEX
            </strong>

            <span>
              RETINAL AI
            </span>

          </div>

        </div>


        <div className="sidebar-section">

          <span className="sidebar-caption">
            WORKSPACE
          </span>

          <nav>

            {[
              ["⌂", "Overview"],
              ["◉", "Screening"],
              ["✦", "Explainability"],
              ["▣", "Reports"],
              ["⌁", "Telemedicine"],
            ].map(
              ([icon, label]) => (

                <button
                  key={label}
                  className={
                    activeNav ===
                    label
                      ? "nav-button active"
                      : "nav-button"
                  }
                  onClick={() =>
                    setActiveNav(
                      label
                    )
                  }
                >

                  <span className="nav-icon">
                    {icon}
                  </span>

                  <span>
                    {label}
                  </span>

                </button>

              )
            )}

          </nav>

        </div>


        <div className="sidebar-bottom">

          <div className="model-status">

            <span className="online-indicator" />

            <div>

              <strong>
                AI Engine Online
              </strong>

              <span>
                Local inference
              </span>

            </div>

          </div>


          <div className="sidebar-version">

            <span>
              RETINEX
            </span>

            <span>
              Prototype v1.0
            </span>

          </div>

        </div>

      </aside>


      {/* ===================================================
          MAIN
          =================================================== */}

      <main className="main-content">

        {/* TOP BAR */}

        <header className="topbar">

          <div>

            <div className="breadcrumb">
              RETINEX
              <span>/</span>
              Screening
            </div>

            <h1>
              Retinal Screening
            </h1>

            <p>
              Explainable AI-assisted diabetic
              retinopathy assessment
            </p>

          </div>


          <div className="topbar-actions">

            <div className="secure-pill">

              <span>
                ✓
              </span>

              Local processing

            </div>

            <div className="avatar">
              AI
            </div>

          </div>

        </header>


        {/* =================================================
            WORKFLOW
            ================================================= */}

        <div className="workflow-line">

          <div className="workflow-step active">
            <span>01</span>
            Image
          </div>

          <div className="workflow-connector" />

          <div
            className={
              quality
                ? "workflow-step active"
                : "workflow-step"
            }
          >
            <span>02</span>
            Quality
          </div>

          <div className="workflow-connector" />

          <div
            className={
              prediction
                ? "workflow-step active"
                : "workflow-step"
            }
          >
            <span>03</span>
            AI Assessment
          </div>

          <div className="workflow-connector" />

          <div
            className={
              prediction?.gradcam_image
                ? "workflow-step active"
                : "workflow-step"
            }
          >
            <span>04</span>
            Explainability
          </div>

          <div className="workflow-connector" />

          <div
            className={
              prediction
                ? "workflow-step active"
                : "workflow-step"
            }
          >
            <span>05</span>
            Report
          </div>

        </div>


        {/* =================================================
            HERO / UPLOAD
            ================================================= */}

        <section className="hero-card">

          <div className="hero-copy">

            <div className="eyebrow">
              AI SCREENING WORKSPACE
            </div>

            <h2>
              Analyze a fundus image
              <br />
              with explainable AI.
            </h2>

            <p>
              Upload a retinal fundus photograph to
              evaluate image quality and run the
              pretrained diabetic retinopathy classifier.
            </p>

            <div className="hero-tags">

              <span>
                DR 0–4 classification
              </span>

              <span>
                Image quality analysis
              </span>

              <span>
                Grad-CAM
              </span>

            </div>

          </div>


          <div className="hero-visual">

            <div className="retina-ring">

              <div className="retina-core">

                <span />

              </div>

            </div>

          </div>

        </section>


        {/* =================================================
            ERROR
            ================================================= */}

        {predictionError && (

          <div className="error-banner">

            <div className="error-icon">
              !
            </div>

            <div>

              <strong>
                AI analysis could not be completed
              </strong>

              <p>
                {predictionError}
              </p>

            </div>

          </div>

        )}


        {/* =================================================
            SCREENING GRID
            ================================================= */}

        <section className="screening-grid">

          {/* =================================================
              IMAGE PANEL
              ================================================= */}

          <div className="card image-card">

            <div className="card-header">

              <div>

                <StepLabel number="01">
                  Fundus Image
                </StepLabel>

                <h3>
                  Retinal image input
                </h3>

              </div>

              <span className="neutral-badge">
                JPG / PNG
              </span>

            </div>


            {!image ? (

              <label className="dropzone">

                <input
                  type="file"
                  accept="image/jpeg,image/png,image/jpg"
                  onChange={
                    handleImage
                  }
                />

                <div className="upload-symbol">
                  ↑
                </div>

                <strong>
                  Upload fundus photograph
                </strong>

                <span>
                  Click to browse or drag an image here
                </span>

                <small>
                  Maximum file size 10 MB
                </small>

                <div className="browse-button">
                  Choose image
                </div>

              </label>

            ) : (

              <div className="image-stage">

                <img
                  src={image}
                  alt="Uploaded retinal fundus"
                />

                <div className="image-overlay">

                  <span>
                    FUNDUS IMAGE
                  </span>

                  <button
                    onClick={
                      changeImage
                    }
                  >
                    Change
                  </button>

                </div>

              </div>

            )}


            <button
              className="primary-action"
              disabled={
                !image ||
                !quality ||
                quality.status ===
                  "RECAPTURE" ||
                screening
              }
              onClick={
                startScreening
              }
            >

              <span>

                {screening
                  ? "Analyzing image..."
                  : prediction
                    ? "Run analysis again"
                    : "Start AI screening"}

              </span>

              <span>
                →
              </span>

            </button>


            {quality?.status ===
              "RECAPTURE" && (

              <div className="recapture-message">

                <strong>
                  Recapture recommended
                </strong>

                <span>
                  {quality.feedback}
                </span>

              </div>

            )}

          </div>


          {/* =================================================
              QUALITY PANEL
              ================================================= */}

          <div className="card quality-card">

            <div className="card-header">

              <div>

                <StepLabel number="02">
                  Image Quality
                </StepLabel>

                <h3>
                  Pre-analysis quality gate
                </h3>

              </div>

              <span
                className={
                  quality
                    ? `status-badge ${quality.status.toLowerCase()}`
                    : "status-badge waiting"
                }
              >
                {quality
                  ? quality.status
                  : "WAITING"}
              </span>

            </div>


            <div className="quality-summary">

              <div className="quality-score">

                <strong>
                  {quality
                    ? quality.overall
                    : "—"}
                </strong>

                <span>
                  /100
                </span>

              </div>


              <div>

                <strong className="quality-title">

                  {!quality

                    ? "Awaiting image"

                    : quality.status ===
                      "GOOD"

                      ? "Suitable for analysis"

                      : quality.status ===
                        "BORDERLINE"

                        ? "Enhancement recommended"

                        : "Recapture required"}

                </strong>

                <p>

                  {quality
                    ? quality.feedback
                    : "Upload a fundus image to evaluate focus, illumination and field of view."}

                </p>

              </div>

            </div>


            <div className="quality-list">

              <MetricBar
                label="Focus / Sharpness"
                value={
                  quality?.focus ??
                  null
                }
              />

              <MetricBar
                label="Illumination"
                value={
                  quality?.illumination ??
                  null
                }
              />

              <MetricBar
                label="Field of View"
                value={
                  quality?.fieldOfView ??
                  null
                }
              />

            </div>


            <div className="quality-note">

              <span>
                i
              </span>

              <p>
                Images rated as ungradeable are blocked
                from AI screening until recaptured.
              </p>

            </div>

          </div>

        </section>


        {/* =================================================
            AI RESULT + GRAD CAM
            ================================================= */}

        <section className="analysis-grid">

          {/* =================================================
              RESULT
              ================================================= */}

          <div className="card result-card">

            <div className="card-header">

              <div>

                <StepLabel number="03">
                  AI Assessment
                </StepLabel>

                <h3>
                  Diabetic retinopathy severity
                </h3>

              </div>

              <span className="ai-pill">
                RESNET50
              </span>

            </div>


            {!prediction ? (

              <div className="empty-result">

                <div className="empty-icon">
                  ✦
                </div>

                <strong>
                  No analysis yet
                </strong>

                <span>
                  Upload a suitable fundus image and
                  start AI screening.
                </span>

              </div>

            ) : (

              <>

                <div className="diagnosis-layout">

                  <div
                    className={`level-display ${
                      levelInfo?.colorClass ||
                      ""
                    }`}
                  >

                    <span>
                      DR LEVEL
                    </span>

                    <strong>
                      {prediction.dr_level}
                    </strong>

                  </div>


                  <div className="diagnosis-copy">

                    <span>
                      International Clinical DR Scale
                    </span>

                    <h2>
                      {levelInfo?.short ||
                        prediction.diagnosis}
                    </h2>

                    <p>
                      {prediction.diagnosis}
                    </p>

                    <div
                      className={
                        prediction.referable
                          ? "referable-pill"
                          : "nonreferable-pill"
                      }
                    >

                      <span>
                        ●
                      </span>

                      {prediction.referable
                        ? "Referable screening result"
                        : "Non-referable screening result"}

                    </div>

                  </div>

                </div>


                <div className="confidence-section">

                  <div className="confidence-top">

                    <span>
                      Model confidence
                    </span>

                    <strong>
                      {prediction.confidence}%
                    </strong>

                  </div>

                  <div className="confidence-track">

                    <div
                      style={{
                        width:
                          `${prediction.confidence}%`,
                      }}
                    />

                  </div>

                </div>


                <div className="result-description">

                  {levelInfo?.description}

                </div>


                {prediction.class_probabilities && (

                  <div className="probabilities">

                    <div className="subsection-title">
                      CLASS PROBABILITIES
                    </div>


                    {Object.entries(
                      prediction.class_probabilities
                    ).map(
                      ([level, value]) => (

                        <div
                          className="probability-row"
                          key={level}
                        >

                          <span>
                            DR {level}
                          </span>

                          <div className="probability-track">

                            <div
                              style={{
                                width:
                                  `${value}%`,
                              }}
                            />

                          </div>

                          <strong>
                            {value}%
                          </strong>

                        </div>

                      )
                    )}

                  </div>

                )}

              </>

            )}

          </div>


          {/* =================================================
              GRAD CAM
              ================================================= */}

          <div className="card explain-card">

            <div className="card-header">

              <div>

                <StepLabel number="04">
                  Explainability
                </StepLabel>

                <h3>
                  Grad-CAM attention map
                </h3>

              </div>

              <span className="xai-pill">
                XAI
              </span>

            </div>


            <div className="gradcam-container">

              {prediction?.gradcam_image ? (

                <img
                  src={
                    prediction.gradcam_image
                  }
                  alt="Grad-CAM visualization showing regions influencing the AI prediction"
                  className="gradcam-image"
                />

              ) : image ? (

                <div className="original-image-container">

                  <img
                    src={image}
                    alt="Retinal fundus awaiting Grad-CAM"
                  />

                  <div className="gradcam-waiting">
                    Run AI screening to generate Grad-CAM
                  </div>

                </div>

              ) : (

                <div className="gradcam-empty">

                  <div>
                    ✦
                  </div>

                  <strong>
                    Explainability preview
                  </strong>

                  <span>
                    Grad-CAM will highlight regions that
                    contributed to the model's prediction.
                  </span>

                </div>

              )}

              {prediction?.gradcam_image && (

                <div className="gradcam-label">
                  GRAD-CAM
                </div>

              )}

            </div>


            <div className="xai-explanation">

              <div className="xai-icon">
                ✦
              </div>

              <div>

                <strong>
                  What does this show?
                </strong>

                <p>
                  Grad-CAM highlights image regions that
                  contributed to the selected classification.
                  It is an explanation of the classifier,
                  not a definitive lesion detector.
                </p>

              </div>

            </div>

          </div>

        </section>


        {/* =================================================
            REPORT
            ================================================= */}

        {prediction && (
  <section className="systemic-section">
    <div className="systemic-heading">
      <div>
        <StepLabel number="06">Systemic Health Awareness</StepLabel>
        <h2>Beyond the retina</h2>
        <p>
          Diabetes can affect blood vessels throughout the body.
          RETINEX provides preventive health guidance alongside the
          retinal screening result.
        </p>
      </div>

      <div className="systemic-disclaimer">
        <span>ⓘ</span>
        <span>Health awareness • Not a diagnosis</span>
      </div>
    </div>

    <div className="systemic-grid">

      {/* KIDNEY */}
      <div className="systemic-card">
        <div className="systemic-card-header">
          <div>
            <span className="systemic-label">
              RENAL SYSTEM
            </span>
            <h3>Kidney Health</h3>
          </div>

          <span className="context-badge kidney-badge">
            Health Awareness
          </span>
        </div>

        <div className="organ-visual kidney-visual">
          <img
            src="/medical/kidney.png"
            alt="Realistic anatomical kidney illustration"
          />
        </div>

        <div className="systemic-content">

          <div className="context-title">
            <div className="context-icon kidney-icon">
              +
            </div>

            <div>
              <strong>Kidney health awareness</strong>

              <p>
                Diabetes can damage the small blood vessels
                that support kidney function.
              </p>
            </div>
          </div>

          <div className="systemic-columns">

            <div className="symptoms-box">
              <span>SYMPTOMS TO WATCH</span>

              <ul>
                <li>Swelling of legs or face</li>
                <li>Foamy urine</li>
                <li>Changes in urination</li>
                <li>Persistent fatigue</li>
                <li>High blood pressure</li>
              </ul>
            </div>

            <div className="solution-box">
              <span>PRECAUTIONS</span>

              <ul>
                <li>Control blood sugar</li>
                <li>Monitor blood pressure</li>
                <li>Regular urine tests</li>
                <li>Check kidney function</li>
                <li>Follow medical advice</li>
              </ul>
            </div>

          </div>
        </div>
      </div>


      {/* HEART */}
      <div className="systemic-card">
        <div className="systemic-card-header">
          <div>
            <span className="systemic-label">
              CARDIOVASCULAR SYSTEM
            </span>
            <h3>Heart Health</h3>
          </div>

          <span className="context-badge heart-badge">
            Health Awareness
          </span>
        </div>

        <div className="organ-visual heart-visual">
          <img
            src="/medical/heart.png"
            alt="Realistic anatomical heart illustration"
          />
        </div>

        <div className="systemic-content">

          <div className="context-title">
            <div className="context-icon heart-icon">
              ♥
            </div>

            <div>
              <strong>Cardiovascular awareness</strong>

              <p>
                Diabetes and vascular disease can increase
                cardiovascular risk.
              </p>
            </div>
          </div>

          <div className="systemic-columns">

            <div className="symptoms-box">
              <span>SYMPTOMS TO WATCH</span>

              <ul>
                <li>Chest discomfort</li>
                <li>Shortness of breath</li>
                <li>Unusual fatigue</li>
                <li>Palpitations</li>
              </ul>
            </div>

            <div className="solution-box">
              <span>PRECAUTIONS</span>

              <ul>
                <li>Control blood sugar</li>
                <li>Monitor blood pressure</li>
                <li>Monitor cholesterol</li>
                <li>Exercise regularly</li>
                <li>Avoid smoking</li>
              </ul>
            </div>

          </div>
        </div>
      </div>

    </div>


    {/* DR-LEVEL GUIDANCE */}
    <div className="severity-context">
      <div className="severity-context-icon">
        ✦
      </div>

      <div>
        <span>RETINEX HEALTH GUIDANCE</span>

        <strong>
          DR Level {prediction.dr_level} — {levelInfo?.short}
        </strong>

        <p>
          {prediction.dr_level === 0 &&
            "Continue regular eye screening and maintain good control of blood sugar, blood pressure and cholesterol."
          }

          {prediction.dr_level === 1 &&
            "Maintain good diabetes management and schedule regular eye and systemic health check-ups."
          }

          {prediction.dr_level === 2 &&
            "Regular eye follow-up and careful management of blood sugar, blood pressure and kidney health are recommended."
          }

          {prediction.dr_level === 3 &&
            "Prompt specialist eye evaluation and close management of diabetes and cardiovascular risk factors are recommended."
          }

          {prediction.dr_level === 4 &&
            "Specialist ophthalmology evaluation should be prioritized. Continue careful management of diabetes and systemic health."
          }
        </p>
      </div>
    </div>


    <div className="systemic-footer">
      <strong>Important:</strong>

      <span>
        RETINEX currently analyzes retinal images for diabetic
        retinopathy. Heart and kidney information is educational
        health awareness based on the relationship between diabetes
        and systemic health. It does not diagnose heart or kidney
        disease.
      </span>
    </div>

  </section>
)}

        {prediction && (
          <section className="report-card">

            <div className="report-heading">

              <div>

                <StepLabel number="05">
                  Screening Report
                </StepLabel>

                <h2>
                  RETINEX AI screening summary
                </h2>

                <p>
                  Generated from the current retinal image
                  analysis.
                </p>

              </div>


              <button
                className="print-button"
                onClick={() =>
                  window.print()
                }
              >
                <span>
                  ⇩
                </span>

                Print / Save
              </button>

            </div>


            <div className="report-content">

              <div className="report-result">

                <span>
                  SCREENING RESULT
                </span>

                <div className="report-result-main">

                  <div
                    className={`report-level ${
                      levelInfo?.colorClass ||
                      ""
                    }`}
                  >
                    {prediction.dr_level}
                  </div>

                  <div>

                    <strong>
                      {prediction.diagnosis}
                    </strong>

                    <span>
                      DR Level {prediction.dr_level}
                      {" • "}
                      {prediction.confidence}% model confidence
                    </span>

                  </div>

                </div>

              </div>


              <div className="report-item">

                <span>
                  REFERABLE STATUS
                </span>

                <strong
                  className={
                    prediction.referable
                      ? "report-warning"
                      : "report-safe"
                  }
                >
                  {prediction.referable
                    ? "Referable"
                    : "Non-referable"}
                </strong>

              </div>


              <div className="report-item">

                <span>
                  IMAGE QUALITY
                </span>

                <strong>
                  {quality
                    ? `${quality.overall}/100`
                    : "—"}
                </strong>

                <small>
                  {quality?.status ||
                    "Not evaluated"}
                </small>

              </div>


              <div className="report-item">

                <span>
                  EXPLAINABILITY
                </span>

                <strong>
                  {prediction.gradcam_image
                    ? "Available"
                    : "Unavailable"}
                </strong>

                <small>
                  Grad-CAM attention map
                </small>

              </div>

            </div>


            <div className="report-recommendation">

              <div className="recommendation-icon">
                !
              </div>

              <div>

                <span>
                  CLINICAL WORKFLOW
                </span>

                <strong>

                  {prediction.referable
                    ? "Ophthalmologist review recommended"
                    : "Clinical review remains recommended"}

                </strong>

                <p>

                  {prediction.referable

                    ? "This prototype classifies the image at a referable DR level. A qualified ophthalmologist should review the image and AI output before any clinical decision."

                    : "The model does not classify this image as referable DR. The AI result should still be reviewed in the appropriate clinical workflow."}

                </p>

              </div>

            </div>


            <div className="report-disclaimer">

              <span>
                RETINEX • AI-assisted screening prototype
              </span>

              <span>
                Not a medical diagnosis
              </span>

            </div>

          </section>

        )}


        {/* =================================================
            FOOTER
            ================================================= */}

        <footer className="footer">

          <div>

            <strong>
              RETINEX
            </strong>

            <span>
              Explainable AI for accessible retinal screening
            </span>

          </div>

          <span>
            Prototype • Human-in-the-loop
          </span>

        </footer>

      </main>

    </div>
  );
}

export default App;