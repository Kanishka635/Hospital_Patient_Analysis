
import { useEffect, useState } from 'react'
import './App.css'

const API = '/api/v1'

function StatCard({ title, value, note }) {
  return (
    <div className="stat-card">
      <p className="muted">{title}</p>
      <h2>{value}</h2>
      <small>{note}</small>
    </div>
  )
}

function App() {
  const [historicalData, setHistoricalData] = useState([])
const [csvError, setCsvError] = useState('')
  const [health, setHealth] = useState('Checking...')
  const [arrivals, setArrivals] = useState([])
  const [summary, setSummary] = useState(null)
  const [error, setError] = useState('')
  const [department, setDepartment] = useState('Emergency')
  const [count, setCount] = useState(12)
  const [saving, setSaving] = useState(false)

  async function loadData() {
    try {
      const healthResponse = await fetch('/health')
      if (!healthResponse.ok) throw new Error('Backend health check failed')
      setHealth('Connected')

      const response = await fetch(`${API}/arrivals`)
      if (!response.ok) throw new Error('Could not load arrivals')
      const data = await response.json()
      const rows = Array.isArray(data)
  ? data
  : data.records ?? data.arrivals ?? data.data ?? []
      setArrivals(rows)

      try {
        const summaryResponse = await fetch(`${API}/summary?period=hour`)
        if (summaryResponse.ok) setSummary(await summaryResponse.json())
      } catch {
        setSummary(null)
      }
      setError('')
    } catch {
      setHealth('Not connected')
      setError('Start your FastAPI backend in a separate terminal. See the instructions below.')
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
  async function loadHistoricalData() {
    try {
      const response = await fetch('/synthetic_hospital_arrivals.csv')
      if (!response.ok) {
        throw new Error('CSV file not found in public folder')
      }

      const csv = await response.text()
      const lines = csv.trim().split(/\r?\n/)
      const headers = lines[0].split(',').map(h => h.trim().toLowerCase())

      const dateIndex = headers.indexOf('date')
      const weekdayIndex = headers.indexOf('weekday')
      const arrivalsIndex = headers.indexOf('arrivals')

      if (dateIndex === -1 || arrivalsIndex === -1) {
        throw new Error('CSV must contain date and arrivals columns')
      }

      const rows = lines.slice(1).map(line => {
        const values = line.split(',')
        return {
          date: values[dateIndex]?.trim(),
          weekday: values[weekdayIndex]?.trim() || '',
          arrivals: Number(values[arrivalsIndex]),
        }
      }).filter(row => row.date && Number.isFinite(row.arrivals))

      setHistoricalData(rows)
      setCsvError('')
    } catch (error) {
      setCsvError(error.message)
    }
  }

  loadHistoricalData()
}, [])

  async function recordArrival(event) {
    event.preventDefault()
    setSaving(true)
    setError('')
    try {
      const response = await fetch(`${API}/arrivals`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        
      body: JSON.stringify({
        period_start: new Date().toISOString(),
        period: 'hour',
        department,
        patient_count: Number(count),
      }),
    })

    const result = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(result.detail || 'Could not save the arrival record. Check the API fields in /docs.')
      }
      await loadData()
      alert('Arrival record submitted. Check the latest data below.')
    } catch (err) {
      setError(err.message || 'Could not save the record.')
    } finally {
      setSaving(false)
    }
  }

  const normalized = arrivals.map((row) => ({
    ...row,
    displayDate: row.period_start || row.timestamp || row.date || row.created_at || row.datetime || '—',
    displayDepartment: row.department || row.ward || 'Emergency',
    displayCount: Number(row.arrivals ?? row.patient_count ?? row.count ?? row.patients ?? 0),
  }))
  const total = normalized.reduce((sum, row) => sum + row.displayCount, 0)
  const average = normalized.length ? total / normalized.length : 0
  const maximum = normalized.length ? Math.max(...normalized.map((row) => row.displayCount)) : 0

  const historicalCounts = historicalData.map(row => row.arrivals)
const historicalTotal = historicalCounts.reduce((sum, n) => sum + n, 0)
const historicalMean = historicalCounts.length
  ? historicalTotal / historicalCounts.length
  : 0

const sortedCounts = [...historicalCounts].sort((a, b) => a - b)

const historicalMedian = sortedCounts.length
  ? sortedCounts.length % 2
    ? sortedCounts[Math.floor(sortedCounts.length / 2)]
    : (sortedCounts[sortedCounts.length / 2 - 1] +
       sortedCounts[sortedCounts.length / 2]) / 2
  : 0

const historicalVariance = historicalCounts.length
  ? historicalCounts.reduce(
      (sum, n) => sum + (n - historicalMean) ** 2, 0
    ) / historicalCounts.length
  : 0

const historicalStdDev = Math.sqrt(historicalVariance)

  return (
    <main className="dashboard">
      <header className="topbar">
        <div>
          <p className="eyebrow">HOSPITAL INTELLIGENCE • STATISTICS PROJECT</p>
          <h1>Patient Arrival Analysis</h1>
          <p className="subtitle">Understand arrivals, explore patterns, and support better planning.</p>
        </div>
        <div className={`connection ${health === 'Connected' ? 'online' : 'offline'}`}>
          <span className="dot" /> API: {health}
        </div>
      </header>

      {error && <div className="notice">{error}</div>}

      <section className="stats-grid">
        <StatCard title="Records loaded" value={normalized.length} note="Arrival observations in API" />
        <StatCard title="Patients in loaded records" value={total.toLocaleString()} note="Sum of loaded arrival counts" />
        <StatCard title="Average per record" value={average.toFixed(1)} note="Descriptive statistic" />
        <StatCard title="Highest record" value={maximum} note="Maximum loaded count" />
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">LIVE API</p>
            <h2>Record patient arrivals</h2>
          </div>
          <button className="secondary" onClick={loadData}>Refresh data</button>
        </div>
        <form className="arrival-form" onSubmit={recordArrival}>
          <label>
            Department
            <select value={department} onChange={(event) => setDepartment(event.target.value)}>
              <option>Emergency</option>
              <option>OPD</option>
              <option>General Ward</option>
              <option>Pediatrics</option>
              <option>ICU</option>
            </select>
          </label>
          <label>
            Patient arrivals
            <input type="number" min="0" max="10000" value={count}
              onChange={(event) => setCount(event.target.value)} required />
          </label>
          <button type="submit" disabled={saving}>{saving ? 'Saving…' : 'Submit record'}</button>
        </form>
        <p className="muted helper">Demo data entry only. Confirm the backend's expected request fields in its API documentation if submission fails.</p>
      </section>

      
<section className="panel">
  <div className="panel-heading">
    <div>
      <p className="eyebrow">HISTORICAL DATASET</p>
      <h2>365-Day Statistical Overview</h2>
    </div>
    <span className="muted">{historicalData.length} observations</span>
  </div>

  {csvError && <div className="notice">{csvError}</div>}

  {!csvError && historicalData.length === 0 && (
    <p className="muted">Loading historical data...</p>
  )}

  {historicalData.length > 0 && (
    <>
      <div className="stats-grid">
        <StatCard title="Total arrivals" value={historicalTotal.toLocaleString()} note="Across historical records" />
        <StatCard title="Mean arrivals" value={historicalMean.toFixed(2)} note="Average per observation" />
        <StatCard title="Median arrivals" value={historicalMedian.toFixed(1)} note="Middle observation" />
        <StatCard title="Standard deviation" value={historicalStdDev.toFixed(2)} note="Population standard deviation" />
      </div>

      <h3>Daily Patient Arrivals</h3>
      <div style={{display: 'flex', alignItems: 'end', gap: '3px', height: '190px', padding: '12px 0', overflowX: 'auto'}}>
        {historicalData.map((row, index) => (
          <div
            key={`${row.date}-${index}`}
            title={`${row.date}: ${row.arrivals} arrivals`}
            style={{
              flex: '1 0 3px',
              minWidth: '3px',
              height: `${Math.max(2, row.arrivals / Math.max(...historicalCounts) * 100)}%`,
              background: '#326de6',
              borderRadius: '3px 3px 0 0'
            }}
          />
        ))}
      </div>
      <p className="muted helper">
        Each bar represents one day's recorded arrivals. Hover over a bar to see its date and count.
      </p>
    </>
  )}
</section>


      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">RECENT OBSERVATIONS</p>
            <h2>Arrival records</h2>
          </div>
          <span className="muted">{normalized.length} records</span>
        </div>
        {normalized.length === 0 ? (
          <div className="empty">
            <h3>No API records found yet</h3>
            <p>Start the backend and refresh. Your existing 365-day CSV analysis is not deleted; this table displays API records only.</p>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead><tr><th>Date / time</th><th>Department</th><th>Arrivals</th></tr></thead>
              <tbody>
                {normalized.slice(-20).reverse().map((row, index) => (
                  <tr key={row.id ?? `${row.displayDate}-${index}`}>
                    <td>{String(row.displayDate)}</td>
                    <td>{row.displayDepartment}</td>
                    <td><strong>{row.displayCount}</strong></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <footer>
        <span>Hospital Patient Arrival Analysis</span>
        <span>Educational demonstration • Verify results against the source dataset</span>
      </footer>
    </main>
  )
}

export default App
