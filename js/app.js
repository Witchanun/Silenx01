let ffmpeg;
let sileroSession;
let selectedFile = null;
let isReady = false;
let sileroState = new Float32Array(2 * 1 * 128);
let speechSegments = [];
let confidenceThreshold = 0.7;
let speechPadding = 200;

const statusText = document.getElementById("status");
const videoInput = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const paddingSlider = document.getElementById("paddingSlider");
const paddingValue = document.getElementById("paddingValue");
const confidenceSlider = document.getElementById("confidenceSlider");
const confidenceValue = document.getElementById("confidenceValue");
const analyzeText = document.getElementById("analyzeText");
const loader = document.getElementById("loader");

let generatedSRT = "";

confidenceSlider.oninput = () => {

    confidenceThreshold =
        Number(confidenceSlider.value);

    confidenceValue.innerText =
        confidenceThreshold;

    console.log(
        "Confidence:",
        confidenceThreshold
    );    

};

paddingSlider.oninput = () => {

    speechPadding = Number(
        paddingSlider.value
    );

    paddingValue.innerText =
        speechPadding;

    console.log(
        "Padding Slider:",
        speechPadding,
        "ms"
    );

};

const { fetchFile } = FFmpeg;

function setStatus(text){

    statusText.innerText = text;

    console.log(text);

}



async function startSilenX(){

    try {


        // Check ONNX Runtime

        if(typeof ort === "undefined"){

            throw new Error("ONNX Runtime not found");

        }

        setStatus("Loading ONNX Runtime...");

        setStatus("✅ ONNX Runtime Loaded");

        setStatus("Loading Silero Model...");


        const modelURL = "model/silero_vad.onnx";


        sileroSession = await ort.InferenceSession.create(
            modelURL
        );

        setStatus("✅ Silero Model Loaded");

        // Load FFmpeg


        setStatus("Loading FFmpeg...");


        if(typeof FFmpeg === "undefined"){

            throw new Error("FFmpeg not found");

        }


        const { createFFmpeg } = FFmpeg;

        ffmpeg = createFFmpeg({
            log: true,
            corePath: "https://cdn.jsdelivr.net/npm/@ffmpeg/core@0.11.0/dist/ffmpeg-core.js"
        });

        await ffmpeg.load();

        setStatus("✅ FFmpeg Loaded");


        setStatus("READY 🚀");
        isReady = true;


    }

    catch(error){

        console.error(error);

        setStatus(
            "ERROR: " + error.message
        );

    }

}

videoInput.addEventListener(
    "change",
    (event) => {

        selectedFile = event.target.files[0];


        if (selectedFile) {
            fileName.innerText = selectedFile.name;
        }

    }
);

async function extractAudio(file) {

    setStatus("Extracting audio...");


    const inputName = "input.mp4";
    const outputName = "audio.wav";


    // ส่งไฟล์เข้า FFmpeg

    ffmpeg.FS(
        "writeFile",
        inputName,
        await fetchFile(file)
    );


    // แปลงเสียง

    await ffmpeg.run(
        "-i",
        inputName,
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        outputName
    );


    // อ่านไฟล์ wav

    const data = ffmpeg.FS(
        "readFile",
        outputName
    );

    setStatus("Audio Ready");

    return data;

}

function wavToFloat32(wavData) {

    const buffer = wavData.buffer;

    const view = new DataView(buffer);


    let offset = 12; // ข้าม RIFF header


    let dataStart = 0;
    let dataSize = 0;


    while (offset < view.byteLength) {

        const chunkId =
            String.fromCharCode(
                view.getUint8(offset),
                view.getUint8(offset + 1),
                view.getUint8(offset + 2),
                view.getUint8(offset + 3)
            );


        const chunkSize =
            view.getUint32(
                offset + 4,
                true
            );


        if (chunkId === "data") {

            dataStart = offset + 8;
            dataSize = chunkSize;

            break;

        }


        offset += 8 + chunkSize;

    }


    console.log(
        "DATA START:",
        dataStart
    );


    console.log(
        "DATA SIZE:",
        dataSize
    );


    const samples =
        new Float32Array(
            dataSize / 2
        );


    let index = 0;


    for (
        let i = dataStart;
        i < dataStart + dataSize;
        i += 2
    ) {


        const sample =
            view.getInt16(
                i,
                true
            );


        samples[index++] =
            sample / 32768;


    }


    return samples;

}

async function runSilero(audio) {

    speechSegments = [];

    sileroState = new Float32Array(2 * 1 * 128);

    let speechStart = null;

    let silenceStart = null;

    const MIN_SILENCE = 0.3;

    const CHUNK_SIZE = 512;

    for (
        let i = 0;
        i < audio.length;
        i += CHUNK_SIZE
    ) {


        const chunk =
            audio.slice(
                i,
                i + CHUNK_SIZE
            );

        const rms =
            Math.sqrt(
                chunk.reduce(
                    (sum, x) => sum + x * x,
                    0
                ) / chunk.length
            );

        // ถ้าช่วงท้ายไม่ครบ 512 ให้ข้ามก่อน

        if (chunk.length !== CHUNK_SIZE) {
            break;
        }



        const inputTensor =
            new ort.Tensor(
                "float32",
                chunk,
                [
                    1,
                    CHUNK_SIZE
                ]
            );



        const stateTensor =
            new ort.Tensor(
                "float32",
                sileroState,
                [
                    2,
                    1,
                    128
                ]
            );



        const srTensor =
            new ort.Tensor(
                "int64",
                BigInt64Array.from([16000n]),
                []
            );



        const results =
            await sileroSession.run({

                input: inputTensor,

                state: stateTensor,

                sr: srTensor

            });



        const prob =
            results.output.data[0];

        console.log(
            "AI Probability:",
            prob,
            "Confidence Threshold:",
            confidenceThreshold
        );



        const time =
            i / 16000;

        console.log(
            "AI Prob:",
            prob,
            "Threshold:",
            confidenceThreshold
        );            

        if (prob > confidenceThreshold) {

            silenceStart = null;

            if(speechStart === null) {

                speechStart = Math.max(
                    0,
                    time - speechPadding / 1000
                );

                console.log(
                    "Padding:",
                    speechPadding,
                    "ms"
                );

                console.log(
                    "Speech Start:",
                    time,
                    "After Padding:",
                    speechStart,
                    "Padding:",
                    speechPadding,
                    "ms"
                );

            }

        }

        else {

            if (speechStart !== null) {

                if (silenceStart === null) {

                    silenceStart = time;

                }


                if (time - silenceStart >= MIN_SILENCE) {

                    speechSegments.push({

                        start: speechStart,

                        end: time + speechPadding / 1000

                    });


                    speechStart = null;
                    silenceStart = null;

                }

            }

        }


        sileroState = new Float32Array(results.stateN.data);


    }

    if (speechStart !== null) {

        speechSegments.push({

            start: speechStart,

            end: audio.length / 16000

        });

    }

}

analyzeBtn.addEventListener(
    "click",
    async () => {

        if (!isReady) {
            alert("System is not ready yet");
            return;
        }

        if (!selectedFile) {
            alert("Please select video");
            return;
        }

        analyzeBtn.classList.add("loading");

        analyzeText.innerText = "Analyzing...";

        analyzeBtn.disabled = true;

        try {
            const audioData = await extractAudio(selectedFile);
            const audioFloat = wavToFloat32(audioData);
            await runSilero(audioFloat);
            console.log(
                "Before Merge:",
                speechSegments
            );

            const finalSegments = speechSegments;

            console.log(
                "Final Segments:",
                finalSegments
            );

            const srtText = generateSRT(finalSegments);

            generatedSRT = srtText;
            downloadBtn.disabled = false;
        }

        catch (error) {
            console.error(error);
            alert(error.message);
        }

        finally {
            analyzeBtn.classList.remove("loading");
            analyzeText.innerText = "Start Analysis";
            analyzeBtn.disabled = false;
        }

    }
);

analyzeBtn.classList.remove("loading");

analyzeText.innerText = "Start Analysis";

analyzeBtn.disabled = false;


function formatSRTTime(seconds) {

    const ms = Math.floor(
        (seconds % 1) * 1000
    );

    const totalSeconds = Math.floor(seconds);

    const hours = Math.floor(totalSeconds / 3600);

    const minutes = Math.floor(
        (totalSeconds % 3600) / 60
    );

    const secs = totalSeconds % 60;


    return (
        String(hours).padStart(2, "0")
        + ":" +
        String(minutes).padStart(2, "0")
        + ":" +
        String(secs).padStart(2, "0")
        + "," +
        String(ms).padStart(3, "0")
    );

}

function generateSRT(segments) {

    let srt = "";


    segments.forEach(
        (segment, index) => {


            srt += index + 1;
            srt += "\n";


            srt +=
                formatSRTTime(segment.start)
                +
                " --> "
                +
                formatSRTTime(segment.end);


            srt += "\n";


            srt += "[Speech]";


            srt += "\n\n";


        }
    );


    return srt;

}



function downloadSRT(text) {

    const blob = new Blob(
        [text],
        {
            type: "text/plain"
        }
    );


    const url = URL.createObjectURL(blob);


    const a = document.createElement("a");

    a.href = url;

    a.download = "SilentCut.srt";

    a.click();

    setTimeout(() => {

        URL.revokeObjectURL(url);

    }, 100);

}

downloadBtn.addEventListener("click", () => {

    if (!generatedSRT) {
        return;
    }

    downloadSRT(generatedSRT);

});


startSilenX();