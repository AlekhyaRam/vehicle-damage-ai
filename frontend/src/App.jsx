import React, { useRef, useState, useEffect } from 'react'
import { predict } from './api'

export default function App() {
  const fileRef = useRef(null)
  const videoRef = useRef(null)
  const [image, setImage] = useState(null)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [buildMesh, setBuildMesh] = useState(false)

  useEffect(() => {
    // Offer camera
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
        }
      } catch {}
    })()
  }, [])

  const capture = () => {
    const video = videoRef.current
    if (!video) return
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)
    canvas.toBlob(b => {
      const file = new File([b], 'capture.jpg', { type: 'image/jpeg' })
      setImage(URL.createObjectURL(b))
      upload(file)
    }, 'image/jpeg', 0.9)
  }

  const onPick = (e) => {
    const f = e.target.files[0]
    if (f) {
      setImage(URL.createObjectURL(f))
      upload(f)
    }
  }

  const upload = async (file) => {
    setBusy(true)
    setResult(null)
    try {
      const data = await predict(file, buildMesh)
      setResult(data)
    } catch (e) {
      alert(e.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div style={{fontFamily:'system-ui, sans-serif', maxWidth:900, margin:'20px auto', padding:'0 12px'}}>
      <h1>Vehicle Damage Detection & Cost Estimation</h1>
      <p>Upload a photo or use your camera. We detect damage using YOLOv8 and estimate repair cost.</p>

      <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:16}}>
        <div style={{border:'1px solid #ddd', borderRadius:12, padding:12}}>
          <h3>Live Camera</h3>
          <video ref={videoRef} autoPlay playsInline style={{width:'100%', borderRadius:8}}/>
          <button onClick={capture} style={{marginTop:8}}>Capture</button>
        </div>

        <div style={{border:'1px solid #ddd', borderRadius:12, padding:12}}>
          <h3>Upload Image</h3>
          <input type="file" accept="image/*" ref={fileRef} onChange={onPick}/>
          <label style={{display:'block', marginTop:8}}>
            <input type="checkbox" checked={buildMesh} onChange={e=>setBuildMesh(e.target.checked)}/> Advanced: build 3D mesh
          </label>
          {image && <img src={image} alt="preview" style={{width:'100%', marginTop:8, borderRadius:8}}/>}
        </div>
      </div>

      <div style={{marginTop:16}}>
        {busy && <p>Running inference…</p>}
        {result && (
          <div style={{border:'1px solid #ddd', borderRadius:12, padding:12}}>
            <h3>Results</h3>
            <pre style={{whiteSpace:'pre-wrap'}}>{JSON.stringify(result, null, 2)}</pre>
            <p><b>Estimated Cost:</b> ₹{result.cost_estimate.estimated_cost.toLocaleString('en-IN')}</p>
            <p><b>Severity Score:</b> {result.cost_estimate.severity_score}</p>
            {result.mesh?.mesh_file && <p>Mesh file generated: {result.mesh.mesh_file}</p>}
          </div>
        )}
      </div>
    </div>
  )
}
