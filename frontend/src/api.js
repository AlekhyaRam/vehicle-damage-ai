export async function predict(file, buildMesh=false) {
  const form = new FormData()
  form.append('file', file)
  const url = `http://localhost:8000/predict?build_mesh=${buildMesh}`
  const res = await fetch(url, { method: 'POST', body: form })
  if (!res.ok) throw new Error('Server error')
  return res.json()
}
