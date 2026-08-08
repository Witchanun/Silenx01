let ffmpeg;
let sileroSession;
let selectedFile = null;
let isReady = false;
let sileroState = new Float32Array(2 * 1 * 128);
let speechSegments = [];
let confidenceThreshold = 0.5;
let speechPadding = 0.02;

const statusText = document.getElementById("status");
const videoInput = document.getElementById("videoInput");
const fileName = document.getElementById("fileName");
const analyzeBtn = document.getElementById("analyzeBtn");
const downloadBtn = document.getElementById("downloadBtn");
const paddingSlider = document.getElementById("paddingSlider");
const paddingValue = document.getElementById("paddingValue");
const analyzeText = document.getElementById("analyzeText");
const loader = document.getElementById("loader");
const fileNameInput = document.getElementById("fileNameInput");

console.log("APP JS LOADED");

paddingSlider.oninput = () => {

    speechPadding = Number(
        paddingSlider.value
    );

    paddingValue.innerText =
        speechPadding.toFixed(2);

    console.log(
        "Padding:",
        speechPadding,
        "seconds"
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

        selectedFile =
            event.target.files[0];

        if (selectedFile) {

            fileName.innerText =
                selectedFile.name;

        }
        else {

            fileName.innerText =
                "No file selected";

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
    let lastSpeechTime = null;
    let silenceStart = null;

    const CHUNK_SIZE = 512;
    const MIN_SILENCE = 0.2;


    for (
        let i = 0;
        i < audio.length;
        i += CHUNK_SIZE
    ) {


        const chunk = audio.slice(
            i,
            i + CHUNK_SIZE
        );


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

        const time =
            i / 16000;

      
        if (prob > confidenceThreshold) {

            if (speechStart === null) {

                speechStart = time;

            }


            lastSpeechTime = time;

            silenceStart = null;


        }

        else {


            if (speechStart !== null) {


                if (silenceStart === null) {

                    silenceStart = time;

                }


                if (
                    time - silenceStart >= MIN_SILENCE
                ) {

                    speechSegments.push({

                        start: speechStart,

                        end: lastSpeechTime

                    });


                    speechStart = null;

                    silenceStart = null;

                    lastSpeechTime = null;

                }

            }

        }


        sileroState =
            new Float32Array(
                results.stateN.data
            );

    }


    if (
        speechStart !== null &&
        lastSpeechTime !== null
    ) {

        speechSegments.push({

            start: speechStart,

            end: lastSpeechTime

        });

    }


    console.log(
        "Raw Segments:",
        speechSegments
    );

}

function mergeOverlappingSegments(segments) {

    if (segments.length === 0) {
        return [];
    }

    const sortedSegments =
        [...segments].sort(
            (a, b) => a.start - b.start
        );

    const merged = [];

    for (const segment of sortedSegments) {

        const last =
            merged[merged.length - 1];

        if (
            last &&
            segment.start <= last.end
        ) {

            last.end =
                Math.max(
                    last.end,
                    segment.end
                );

        }
        else {

            merged.push({
                start: segment.start,
                end: segment.end
            });

        }

    }

    return merged;
}

function applyPadding(segments) {

    return segments.map(segment => ({

        start: Math.max(
            0,
            segment.start - speechPadding
        ),

        end:
            segment.end + speechPadding

    }));

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
                "Raw Segments:",
                speechSegments
            );

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


    const url =
        URL.createObjectURL(blob);


    const a =
        document.createElement("a");


    a.href = url;

    let fileName =
        fileNameInput.value.trim();

    if (fileName === "") {
        fileName = "SilenX";
    }

    if (!fileName.toLowerCase().endsWith(".srt")) {
        fileName += ".srt";
    }

    a.download = fileName;

    a.click();


    URL.revokeObjectURL(url);

}

downloadBtn.addEventListener(
    "click",
    () => {

        if (speechSegments.length === 0) {
            return;
        }


        const paddedSegments =
            applyPadding(
                speechSegments
            );

        const finalSegments =
            mergeOverlappingSegments(
                paddedSegments
            );


        const srtText =
            generateSRT(
                finalSegments
            );


        downloadSRT(
            srtText
        );

    }
);

startSilenX();