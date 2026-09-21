import React from 'react';
import { createRoot } from 'react-dom/client';
import { Activity, AlertTriangle, BarChart3, CalendarClock, Download, FileText, HeartPulse, Home, Loader2, MessageSquare, Send, Settings, ShieldPlus, Stethoscope, Upload, User, UserPlus, Users } from 'lucide-react';
import './styles.css';

const API_URL = import.meta.env.VITE_API_URL || '';

const initialPredictionVariables = {
  espace_PR: 160,
  cause_valvulaire: 0,
  PAD: 65,
  PAS: 110,
  OG: 22,
  Uree: 7,
  statine: 1,
  ATCD_d_hospitalisation: 0,
  IMC: 24,
  IEC_dose: 10,
  FQ_ECG_sortie: 75,
  HTAP: 35,
  TP: 75,
  lymphocyte: 1.2,
  ARM: 0,
  QT_corrige: 420,
  Glycemie_a_jeun: 1.0,
  ProBNP: 1500,
  dose_de_lasilix: 40,
};

const initialOrdonnanceVariables = {
  lasilix: 1,
  betabloquants: 0,
  dose_BB_pourcentage: '',
  plavix_cardiocine: 0,
  sintrome_aod: 0,
  ARM: 0,
  INH_SGLT2: 0,
  Ivabradine: 0,
};

const initialVariables = { ...initialPredictionVariables, ...initialOrdonnanceVariables };

const emptyPredictionVariables = {
  espace_PR: '',
  cause_valvulaire: '',
  PAD: '',
  PAS: '',
  OG: '',
  Uree: '',
  statine: '',
  ATCD_d_hospitalisation: '',
  IMC: '',
  IEC_dose: '',
  FQ_ECG_sortie: '',
  HTAP: '',
  TP: '',
  lymphocyte: '',
  ARM: '',
  QT_corrige: '',
  Glycemie_a_jeun: '',
  ProBNP: '',
  dose_de_lasilix: '',
};

const emptyOrdonnanceVariables = {
  lasilix: '',
  betabloquants: '',
  dose_BB_pourcentage: '',
  plavix_cardiocine: '',
  sintrome_aod: '',
  ARM: '',
  INH_SGLT2: '',
  Ivabradine: '',
};

const emptyVariables = { ...emptyPredictionVariables, ...emptyOrdonnanceVariables };

const initialIdentity = { nom: '', prenom: '', date_naissance: '', sexe: '' };

const fields = [
  ['espace_PR', 'Espace PR', 'ms'],
  ['cause_valvulaire', 'Cause valvulaire', '0/1'],
  ['PAD', 'PAD', 'mmHg'],
  ['PAS', 'PAS', 'mmHg'],
  ['OG', 'Oreillette gauche', 'cm²'],
  ['Uree', 'Urée', 'mmol/L'],
  ['statine', 'Statine', '0/1'],
  ['ATCD_d_hospitalisation', 'ATCD hospitalisation', 'nombre'],
  ['IMC', 'IMC', 'kg/m2'],
  ['IEC_dose', 'Dose IEC', 'mg/j'],
  ['FQ_ECG_sortie', 'FC sortie', 'bpm'],
  ['HTAP', 'HTAP', 'mmHg'],
  ['TP', 'TP', '%'],
  ['lymphocyte', 'Lymphocytes', 'G/L'],
  ['ARM', 'ARM', '0/1'],
  ['QT_corrige', 'QT corrigé', 'ms'],
  ['Glycemie_a_jeun', 'Glycémie à jeun', 'g/L'],
  ['ProBNP', 'BNP', 'pg/mL'],
  ['dose_de_lasilix', 'Dose Lasilix', 'mg'],
];

const ordonnanceSelects = [
  ['lasilix', 'Lasilix', [['', '--'], [0, 'Non'], [1, 'Oui']]],
  ['betabloquants', 'Betabloquant', [['', '--'], [0, 'Non'], [1, 'Oui']]],
  ['plavix_cardiocine', 'Plavix / Cardiocine', [['', '--'], [0, 'Aucun'], [1, 'Plavix'], [2, 'Cardiocine 100']]],
  ['sintrome_aod', 'Sintrom / AOD', [['', '--'], [0, 'Aucun'], [1, 'Sintrom'], [2, 'AOD']]],
  ['ARM', 'ARM', [['', '--'], [0, 'Non'], [1, 'Oui']]],
  ['INH_SGLT2', 'ISGLT2', [['', '--'], [0, 'Non'], [1, 'Oui']]],
  ['Ivabradine', 'Ivabradine', [['', '--'], [0, 'Non'], [1, 'Oui']]],
];

const predictionKeys = fields.map(([key]) => key);

function predictionVariables(values) {
  const out = Object.fromEntries(predictionKeys.map((key) => [key, values[key]]));
  const dose = Number(values.dose_de_lasilix);
  out.dose_lasilix_sup120 = dose >= 120 ? 1 : 0;
  delete out.dose_de_lasilix;
  const htapVal = Number(values.HTAP);
  out.HTAP = htapVal < 40 ? 0 : htapVal < 60 ? 1 : 2;
  return out;
}

function saveVariables(values) {
  return Object.fromEntries(Object.entries(values).map(([key, value]) => [key, value === '' ? null : value]));
}

function withVariableDefaults(values = {}) {
  const merged = { ...initialVariables, ...values };
  if (merged.plavix_cardiocine === undefined) {
    merged.plavix_cardiocine = Number(merged.plavix) === 1 ? 1 : Number(merged.cardiocine100) === 1 ? 2 : 0;
  }
  if (merged.sintrome_aod === undefined) {
    merged.sintrome_aod = Number(merged.sintrom) === 1 ? 1 : Number(merged.AOD) === 1 ? 2 : 0;
  }
  return merged;
}

function riskColor(level) {
  if (level === 'faible') return '#16a34a';
  if (level === 'modere') return '#f59e0b';
  if (level === 'eleve') return '#ef4444';
  return '#991b1b';
}

function calculateAge(dateNaissance) {
  if (!dateNaissance) return '--';
  const birth = new Date(dateNaissance);
  if (Number.isNaN(birth.getTime())) return '--';
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const monthDiff = today.getMonth() - birth.getMonth();
  if (monthDiff < 0 || (monthDiff === 0 && today.getDate() < birth.getDate())) age -= 1;
  return age;
}

function Gauge({ score, level }) {
  const deg = Math.min(180, Math.max(0, score * 1.8));
  return (
    <div className="gaugeCard">
      <div className="gauge" style={{ '--deg': `${deg}deg`, '--risk': riskColor(level) }}>
        <div className="needle" />
        <div className="gaugeInner"><strong>{score.toFixed(1)}%</strong><span>Score de risque</span></div>
      </div>
      <div className="riskBadge" style={{ background: riskColor(level) }}>Niveau {level}</div>
      <p>Probabilité estimée de réhospitalisation à 3 mois.</p>
    </div>
  );
}

function App() {
  // toujours afficher login à l'ouverture (pas d'auto-login direct)
  const [doctor, setDoctor] = React.useState(null);
  const [patientAuth, setPatientAuth] = React.useState(null);
  const [space, setSpace] = React.useState('selector');
  // nettoyer toute session persistante au chargement
  React.useEffect(() => {
    localStorage.removeItem('icfer_doctor');
    localStorage.removeItem('icfer_patient');
  }, []);
  const [authMode, setAuthMode] = React.useState('login');
  const [loginForm, setLoginForm] = React.useState({ name: '', password: '' });
  // patient auth forms
  const [patientMode, setPatientMode] = React.useState('login');
  const [patientForm, setPatientForm] = React.useState({ email: '', password: '', nom: '', prenom: '', doctor_id: '', date_naissance: '', sexe: '' });
  const [doctorsList, setDoctorsList] = React.useState([]);
  const [view, setView] = React.useState('dashboard');
  const [patients, setPatients] = React.useState([]);
  const [criticalPatients, setCriticalPatients] = React.useState([]);
  const [selectedPatient, setSelectedPatient] = React.useState(null);
  const [identity, setIdentity] = React.useState(initialIdentity);
  const [variables, setVariables] = React.useState(initialVariables);
  const [settingsForm, setSettingsForm] = React.useState({ name: '', password: '', photo_url: '' });
  const [importStatus, setImportStatus] = React.useState('');
  const [result, setResult] = React.useState(null);
  const [showOrdonnance, setShowOrdonnance] = React.useState(false);
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState('');

  // patient messaging state
  const [messages, setMessages] = React.useState([]);
  const [newMsg, setNewMsg] = React.useState('');
  const [bilans, setBilans] = React.useState([]);
  const [uploadDesc, setUploadDesc] = React.useState('');
  const [uploadFile, setUploadFile] = React.useState(null);
  // doctor messaging state
  const [conversations, setConversations] = React.useState([]);
  const [selectedConv, setSelectedConv] = React.useState(null);
  const [doctorReply, setDoctorReply] = React.useState('');
  const [bilanReply, setBilanReply] = React.useState({});

  const currentDoctorId = doctor ? doctor.id : 'default';

  const loadDoctorsList = React.useCallback(async () => {
    try {
      const res = await fetch(`${API_URL}/api/doctors/list`);
      const data = await res.json();
      setDoctorsList(data.doctors || []);
      if (data.doctors?.length && !patientForm.doctor_id) {
        setPatientForm((p) => ({ ...p, doctor_id: data.doctors[0].id }));
      }
    } catch {}
  }, []);

  React.useEffect(() => { loadDoctorsList(); }, [loadDoctorsList]);

  const loadPatients = React.useCallback(async () => {
    if (!doctor) return;
    const [allRes, criticalRes] = await Promise.all([
      fetch(`${API_URL}/api/patients?doctor_id=${currentDoctorId}`),
      fetch(`${API_URL}/api/patients/critical?doctor_id=${currentDoctorId}`),
    ]);
    const all = await allRes.json();
    const critical = await criticalRes.json();
    setPatients(all.patients || []);
    setCriticalPatients(critical.patients || []);
  }, [doctor, currentDoctorId]);

  const loadPatientMessages = React.useCallback(async () => {
    if (!patientAuth) return;
    const [msgRes, bilanRes] = await Promise.all([
      fetch(`${API_URL}/api/messages?patient_auth_id=${patientAuth.id}&doctor_id=${patientAuth.doctor_id}`),
      fetch(`${API_URL}/api/bilans?patient_auth_id=${patientAuth.id}&doctor_id=${patientAuth.doctor_id}`),
    ]);
    const m = await msgRes.json();
    const b = await bilanRes.json();
    setMessages(m.messages || []);
    setBilans(b.bilans || []);
  }, [patientAuth]);

  const loadConversations = React.useCallback(async () => {
    if (!doctor) return;
    const res = await fetch(`${API_URL}/api/messages/doctor?doctor_id=${currentDoctorId}`);
    const data = await res.json();
    setConversations(data.conversations || []);
  }, [doctor, currentDoctorId]);

  React.useEffect(() => { loadPatients().catch(() => {}); }, [loadPatients]);
  React.useEffect(() => { if (patientAuth) loadPatientMessages().catch(()=>{}); }, [loadPatientMessages]);
  React.useEffect(() => { if (doctor) loadConversations().catch(()=>{}); }, [loadConversations]);
  React.useEffect(() => {
    if (doctor) setSettingsForm({ name: doctor.name || '', password: '', photo_url: doctor.photo_url || '' });
  }, [doctor]);

  // polling for messages
  React.useEffect(() => {
    if (!patientAuth && !doctor) return;
    const id = setInterval(() => {
      if (patientAuth) loadPatientMessages().catch(()=>{});
      if (doctor) loadConversations().catch(()=>{});
    }, 8000);
    return () => clearInterval(id);
  }, [patientAuth, doctor, loadPatientMessages, loadConversations]);

  const authenticate = async () => {
    setError('');
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/doctors/${authMode === 'login' ? 'login' : 'signup'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(loginForm),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Erreur de connexion');
      localStorage.setItem('icfer_doctor', JSON.stringify(data.doctor));
      localStorage.removeItem('icfer_patient');
      setDoctor(data.doctor);
      setPatientAuth(null);
      setSpace('doctor');
    } catch (e) {
      setError(e.message || 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  const authenticatePatient = async () => {
    setError('');
    setLoading(true);
    try {
      const url = patientMode === 'login' ? `${API_URL}/api/patient/login` : `${API_URL}/api/patient/signup`;
      const body = patientMode === 'login' ? { email: patientForm.email, password: patientForm.password } : patientForm;
      const res = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erreur patient');
      localStorage.setItem('icfer_patient', JSON.stringify(data.patient));
      localStorage.removeItem('icfer_doctor');
      setPatientAuth(data.patient);
      setDoctor(null);
      setSpace('patient');
    } catch (e) {
      setError(e.message || 'Erreur inconnue');
    } finally { setLoading(false); }
  };

  const logout = () => {
    localStorage.removeItem('icfer_doctor');
    setDoctor(null);
    setView('dashboard');
    setSpace('selector');
  };
  const logoutPatient = () => {
    localStorage.removeItem('icfer_patient');
    setPatientAuth(null);
    setSpace('selector');
  };

  const sendPatientMessage = async () => {
    if (!newMsg.trim() || !patientAuth) return;
    setLoading(true);
    try {
      await fetch(`${API_URL}/api/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ patient_auth_id: patientAuth.id, doctor_id: patientAuth.doctor_id, sender_role: 'patient', content: newMsg }) });
      setNewMsg('');
      await loadPatientMessages();
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const sendDoctorMessage = async () => {
    if (!doctorReply.trim() || !selectedConv) return;
    setLoading(true);
    try {
      await fetch(`${API_URL}/api/messages`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ patient_auth_id: selectedConv.patient_auth_id, doctor_id: currentDoctorId, sender_role: 'doctor', content: doctorReply }) });
      setDoctorReply('');
      await loadConversations();
      // refresh selected
      const res = await fetch(`${API_URL}/api/messages?patient_auth_id=${selectedConv.patient_auth_id}&doctor_id=${currentDoctorId}`);
      const data = await res.json();
      setSelectedConv((prev) => ({ ...prev, messages: data.messages }));
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const uploadBilan = async () => {
    if (!uploadFile || !patientAuth) { setError('Choisissez un fichier PDF/JPG/PNG'); return; }
    setLoading(true); setError('');
    try {
      const fd = new FormData();
      fd.append('patient_auth_id', patientAuth.id);
      fd.append('doctor_id', patientAuth.doctor_id);
      fd.append('description', uploadDesc);
      fd.append('file', uploadFile);
      const res = await fetch(`${API_URL}/api/bilans/upload`, { method: 'POST', body: fd });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erreur upload');
      setUploadFile(null); setUploadDesc('');
      await loadPatientMessages();
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const replyBilan = async (bilanId) => {
    const reply = bilanReply[bilanId];
    if (!reply?.trim()) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/bilans/${bilanId}/reply`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doctor_id: currentDoctorId, reply }) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Erreur réponse');
      setBilanReply((p) => ({ ...p, [bilanId]: '' }));
      await loadConversations();
    } catch (e) { setError(e.message); } finally { setLoading(false); }
  };

  const startNewPatient = () => {
    setSelectedPatient(null);
    setIdentity(initialIdentity);
    setVariables(emptyVariables);
    setResult(null);
    setShowOrdonnance(false);
    setView('add');
  };

  const openPatient = (patient) => {
    const lastConsultation = patient.consultations?.[patient.consultations.length - 1];
    setSelectedPatient(patient);
    setIdentity({
      nom: patient.nom || '',
      prenom: patient.prenom || '',
      date_naissance: patient.date_naissance || '',
      sexe: patient.sexe || '',
    });
    setVariables(withVariableDefaults(lastConsultation?.variables));
    setResult(null);
    setShowOrdonnance(false);
    setView('detail');
  };

  const submit = async (withReport = false) => {
    setError('');
    if (!identity.nom || !identity.prenom || !identity.date_naissance || !identity.sexe) {
      setError('Veuillez remplir nom, prénom, date de naissance et sexe.');
      return;
    }
    const missingField = fields.find(([key]) => variables[key] === '');
    if (missingField) {
      setError(`Veuillez remplir le champ: ${missingField[1]}.`);
      return;
    }
    setLoading(true);
    try {
      const predictRes = await fetch(`${API_URL}/api/${withReport ? 'report' : 'predict'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(withReport ? saveVariables(variables) : predictionVariables(variables)),
      });
      if (!predictRes.ok) throw new Error(`Erreur API ${predictRes.status}`);
      const data = await predictRes.json();
      setResult(data);
      const saveBody = {
        patient_id: selectedPatient?.id || null,
        doctor_id: currentDoctorId,
        ...identity,
        ...saveVariables(variables),
      };
      const saveRes = await fetch(`${API_URL}/api/patients`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(saveBody),
      });
      const saved = await saveRes.json();
      setSelectedPatient(saved.record);
      await loadPatients();
    } catch (e) {
      setError(e.message || 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  const openConsultation = async (consultation) => {
    setError('');
    setLoading(true);
    try {
      const consultationVariables = withVariableDefaults(consultation.variables);
      setVariables(consultationVariables);
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(predictionVariables(consultationVariables)),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Erreur API ${response.status}`);
      setResult(data);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (e) {
      setError(e.message || 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  const updateProfilePhoto = (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => setSettingsForm((p) => ({ ...p, photo_url: reader.result }));
    reader.readAsDataURL(file);
  };

  const saveSettings = async () => {
    setError('');
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/doctors/profile`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ doctor_id: currentDoctorId, ...settingsForm }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Erreur paramètres');
      localStorage.setItem('icfer_doctor', JSON.stringify(data.doctor));
      setDoctor(data.doctor);
      setSettingsForm((p) => ({ ...p, password: '' }));
    } catch (e) {
      setError(e.message || 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  const importPatients = async (file) => {
    if (!file) return;
    setError('');
    setImportStatus('Import en cours...');
    setLoading(true);
    try {
      const contentBase64 = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(file);
      });
      const response = await fetch(`${API_URL}/api/patients/import?doctor_id=${currentDoctorId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, content_base64: contentBase64 }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Erreur import');
      setImportStatus(`${data.imported} patient(s) importé(s). ${data.errors?.length ? `${data.errors.length} ligne(s) avec erreur.` : ''}`);
      await loadPatients();
    } catch (e) {
      setError(e.message || 'Erreur inconnue');
      setImportStatus('');
    } finally {
      setLoading(false);
    }
  };

  const exportPatients = () => {
    window.location.href = `${API_URL}/api/patients/export?doctor_id=${currentDoctorId}`;
  };

  // ── Rendu : sélecteur si aucun connecté ──
  if (!doctor && !patientAuth) {
    return (
      <div className="loginScreen">
        <div style={{ display: 'grid', gap: 18, width: 'min(760px,100%)' }}>
          <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
            <button className="primaryBtn" onClick={()=>setSpace('selector')}>Accueil</button>
          </div>

          {space==='selector' && (
            <div className="panel" style={{ textAlign:'center', padding: 32 }}>
              <h1 style={{ margin:'0 0 6px', fontSize:26, color:'#0f172a' }}>Bienvenue sur CardioRisk AI</h1>
              <p style={{ margin:'0 0 18px', color:'#64748b', fontSize:14 }}>Votre assistant intelligent pour anticiper les risques cardiaques et faciliter la communication médecin-patient</p>
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginTop:14 }}>
                <button onClick={()=>setSpace('doctor')} style={{ cursor:'pointer', background:'linear-gradient(135deg,#4f46e5,#06b6d4)', color:'white', border:'0', borderRadius:14, padding:'16px 14px', display:'grid', gap:4, textAlign:'left' }}>
                  <span style={{ display:'flex', alignItems:'center', gap:8, fontWeight:800, fontSize:14 }}><Stethoscope size={18}/> Espace Médecin →</span>
                  <small style={{ opacity:.9, fontSize:11 }}>Tableau de bord, calcul IA, suivi consultations</small>
                </button>
                <button onClick={()=>{setSpace('patient'); loadDoctorsList();}} style={{ cursor:'pointer', background:'white', color:'#1e293b', border:'1px solid #cbd5e1', borderRadius:14, padding:'16px 14px', display:'grid', gap:4, textAlign:'left' }}>
                  <span style={{ display:'flex', alignItems:'center', gap:8, fontWeight:800, fontSize:14 }}><User size={18} style={{ color:'#16a34a' }}/> Espace Patient →</span>
                  <small style={{ color:'#64748b', fontSize:11 }}>Accédez à votre suivi, messages et bilans</small>
                </button>
              </div>
            </div>
          )}

          {space==='doctor' && (
            <div className="loginCard" style={{ margin:'0 auto' }}>
              <div className="brand loginBrand"><HeartPulse size={28} /><div><b>CardioRisk AI</b><span>Espace cardiologue</span></div></div>
              <h1>{authMode === 'login' ? 'Connexion cardiologue' : 'Inscription cardiologue'}</h1>
              <p>Chaque compte voit uniquement ses propres patients.</p>
              {error && <div className="error">{error}</div>}
              <label><span>Nom du médecin</span><input value={loginForm.name} onChange={(e) => setLoginForm((p) => ({ ...p, name: e.target.value }))} /></label>
              <label><span>Mot de passe</span><input type="password" value={loginForm.password} onChange={(e) => setLoginForm((p) => ({ ...p, password: e.target.value }))} /></label>
              <button className="primaryBtn" onClick={authenticate} disabled={loading}>{loading ? <Loader2 className="spin" size={18} /> : null}{authMode === 'login' ? 'Login' : 'Sign in'}</button>
              <button className="ghostBtn loginSwitch" onClick={() => setAuthMode((m) => (m === 'login' ? 'signup' : 'login'))}>{authMode === 'login' ? 'Créer un compte' : 'J’ai déjà un compte'}</button>
              <button className="ghostBtn loginSwitch" onClick={()=>setSpace('selector')}>← Retour</button>
            </div>
          )}

          {space==='patient' && (
            <div className="loginCard" style={{ margin:'0 auto' }}>
              <div className="brand loginBrand"><HeartPulse size={28} /><div><b>CardioRisk AI</b><span>Espace patient</span></div></div>
              <h1>{patientMode === 'login' ? 'Connexion patient' : 'Inscription patient'}</h1>
              <p>Accédez à votre messagerie et envoyez vos bilans à votre cardiologue.</p>
              {error && <div className="error">{error}</div>}
              {patientMode==='signup' && (
                <>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
                    <label><span>Nom</span><input value={patientForm.nom} onChange={(e)=>setPatientForm(p=>({...p, nom:e.target.value}))} /></label>
                    <label><span>Prénom</span><input value={patientForm.prenom} onChange={(e)=>setPatientForm(p=>({...p, prenom:e.target.value}))} /></label>
                  </div>
                  <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12 }}>
                    <label><span>Date naissance</span><input type="date" value={patientForm.date_naissance} onChange={(e)=>setPatientForm(p=>({...p, date_naissance:e.target.value}))} /></label>
                    <label><span>Sexe</span><select value={patientForm.sexe} onChange={(e)=>setPatientForm(p=>({...p, sexe:e.target.value}))}><option value="">--</option><option value="M">M</option><option value="F">F</option></select></label>
                  </div>
                  <label><span>Médecin traitant</span><select value={patientForm.doctor_id} onChange={(e)=>setPatientForm(p=>({...p, doctor_id:e.target.value}))}>{doctorsList.map(d=><option key={d.id} value={d.id}>{d.name}</option>)}{doctorsList.length===0 && <option value="">Aucun médecin</option>}</select></label>
                </>
              )}
              <label><span>Email</span><input value={patientForm.email} onChange={(e)=>setPatientForm(p=>({...p, email:e.target.value}))} placeholder="patient@email.com" /></label>
              <label><span>Mot de passe</span><input type="password" value={patientForm.password} onChange={(e)=>setPatientForm(p=>({...p, password:e.target.value}))} /></label>
              {patientMode==='signup' && !patientForm.doctor_id && <p className="muted">Choisissez votre cardiologue</p>}
              <button className="primaryBtn" onClick={authenticatePatient} disabled={loading}>{loading ? <Loader2 className="spin" size={18}/> : null}{patientMode==='login'?'Connexion':'Créer mon compte'}</button>
              <button className="ghostBtn loginSwitch" onClick={()=>setPatientMode(m=>m==='login'?'signup':'login')}>{patientMode==='login'?"Créer un compte patient":"J'ai déjà un compte"}</button>
              <button className="ghostBtn loginSwitch" onClick={()=>setSpace('selector')}>← Retour</button>
            </div>
          )}
        </div>
      </div>
    );
  }

  // ── Espace Patient connecté ──
  if (patientAuth) {
    const doctorName = doctorsList.find(d=>d.id===patientAuth.doctor_id)?.name || patientAuth.doctor_id;
    return (
      <div className="appShell" style={{ gridTemplateColumns:'1fr' }}>
        <main style={{ maxWidth: 900, margin:'0 auto', width:'100%' }}>
          <header>
            <div><h1>Espace Patient</h1><p>Bienvenue {patientAuth.prenom} {patientAuth.nom} — Médecin: {doctorName}</p></div>
            <div className="doctorBar">
              <div className="doctorChip"><User size={18}/><div><b>{patientAuth.prenom} {patientAuth.nom}</b><small>{patientAuth.email}</small></div></div>
              <button className="ghostBtn noMargin" onClick={logoutPatient}>Déconnexion</button>
            </div>
          </header>
          {error && <div className="error">{error}</div>}

          <section className="panel" style={{ marginBottom:16 }}>
            <h2><Upload size={18}/> Envoyer un bilan à votre médecin</h2>
            <p className="muted">Formats acceptés: PDF, PNG, JPG — max 10 Mo. Votre médecin sera notifié et pourra répondre.</p>
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:12, marginTop:12 }}>
              <label><span>Description (optionnel)</span><input placeholder="Ex: Bilan sanguin 21/09, ECG..." value={uploadDesc} onChange={e=>setUploadDesc(e.target.value)} /></label>
              <label><span>Fichier</span><input type="file" accept=".pdf,.png,.jpg,.jpeg" onChange={e=>setUploadFile(e.target.files?.[0]||null)} /></label>
            </div>
            <button className="primaryBtn" onClick={uploadBilan} disabled={loading} style={{ marginTop:12 }}>{loading?<Loader2 className="spin" size={18}/>:<Upload size={18}/>} Envoyer le bilan</button>
            {uploadFile && <div className="successMsg" style={{ marginTop:10 }}>Fichier prêt: {uploadFile.name}</div>}
          </section>

          <div style={{ display:'grid', gridTemplateColumns:'1.2fr .8fr', gap:16 }}>
            <section className="panel">
              <h2><MessageSquare size={18}/> Conversation avec {doctorName}</h2>
              <div style={{ maxHeight: 420, overflowY:'auto', display:'grid', gap:8, background:'#f8fafc', padding:12, borderRadius:12, border:'1px solid #e5e7eb' }}>
                {messages.length===0 && <p className="muted">Aucun message. Envoyez votre premier message ou bilan.</p>}
                {messages.map(m=>(
                  <div key={m.id} style={{ justifySelf: m.sender_role==='patient'?'end':'start', background: m.sender_role==='patient' ? '#4f46e5' : 'white', color: m.sender_role==='patient' ? 'white':'#1e293b', padding:'10px 12px', borderRadius:12, maxWidth:'78%', border:'1px solid #e5e7eb', fontSize:13 }}>
                    <div style={{ whiteSpace:'pre-wrap' }}>{m.content}</div>
                    <small style={{ opacity:.7, fontSize:11 }}>{new Date(m.created_at).toLocaleString()}</small>
                  </div>
                ))}
              </div>
              <div style={{ display:'flex', gap:8, marginTop:12 }}>
                <input style={{ flex:1 }} placeholder="Écrivez votre message..." value={newMsg} onChange={e=>setNewMsg(e.target.value)} onKeyDown={e=>e.key==='Enter' && sendPatientMessage()} />
                <button className="primaryBtn" onClick={sendPatientMessage} disabled={loading}><Send size={18}/> Envoyer</button>
              </div>
            </section>

            <section className="panel">
              <h2><FileText size={18}/> Mes bilans</h2>
              <div style={{ display:'grid', gap:10, maxHeight: 520, overflowY:'auto' }}>
                {bilans.length===0 && <p className="muted">Aucun bilan envoyé.</p>}
                {bilans.map(b=>(
                  <div key={b.id} style={{ background:'#f8fafc', border:'1px solid #e5e7eb', borderRadius:12, padding:12 }}>
                    <b style={{ fontSize:13 }}>{b.filename}</b><br/>
                    <small className="muted">{b.description || 'Sans description'} — {new Date(b.created_at).toLocaleDateString()} — {b.status==='repondu'?'✅ Répondu':'⏳ En attente'}</small>
                    <div style={{ display:'flex', gap:8, marginTop:8 }}>
                      <a className="ghostBtn noMargin" style={{ fontSize:12, padding:'6px 10px', textDecoration:'none' }} href={`${API_URL}/api/bilans/download/${b.id}`} target="_blank" rel="noreferrer">Voir / Télécharger</a>
                    </div>
                    {b.reply && <div style={{ marginTop:8, background:'#dcfce7', border:'1px solid #bbf7d0', borderRadius:8, padding:8, fontSize:12 }}><b>Réponse médecin:</b> {b.reply}</div>}
                  </div>
                ))}
              </div>
            </section>
          </div>
        </main>
      </div>
    );
  }

  const activePatient = selectedPatient || { ...identity, id: 'nouveau' };
  const activeRisk = result || selectedPatient?.last_result;
  const totalConsultations = patients.reduce((sum, p) => sum + (p.consultations?.length || 0), 0);
  const averageScore = patients.length
    ? patients.reduce((sum, p) => sum + (p.last_result?.score_percent || 0), 0) / patients.length
    : 0;
  const riskCounts = patients.reduce((acc, p) => {
    const level = p.last_result?.risk_level || 'non calcule';
    acc[level] = (acc[level] || 0) + 1;
    return acc;
  }, {});

  return (
    <div className="appShell">
      <aside>
        <div className="brand"><HeartPulse size={26} /> <div><b>CardioRisk AI</b><span>Prédiction et suivi<br />cardiologique</span></div></div>
        <nav>
          <button className={view === 'dashboard' ? 'active' : ''} onClick={() => setView('dashboard')}><Home size={18} /> Tableau de bord</button>
          <button className={view === 'patients' ? 'active' : ''} onClick={() => setView('patients')}><Users size={18} /> Mes patients</button>
          {(() => { const unread = conversations.reduce((s,c)=> s + (c.unread_count||0), 0); return <button className={view === 'messagerie' ? 'active' : ''} onClick={() => { setView('messagerie'); loadConversations(); }}><MessageSquare size={18} /> Messagerie / Bilans {unread>0 && <span style={{ background:'#ef4444', color:'white', borderRadius:999, padding:'2px 7px', fontSize:11 }}>{unread}</span>}</button>; })()}
          <button className={view === 'importExport' ? 'active' : ''} onClick={() => setView('importExport')}><Upload size={18} /> Importer / Exporter</button>
          <button className={view === 'statistics' ? 'active' : ''} onClick={() => setView('statistics')}><BarChart3 size={18} /> Statistiques</button>
          <button className={view === 'settings' ? 'active' : ''} onClick={() => setView('settings')}><Settings size={18} /> Paramètres</button>
        </nav>
        <button className="addPatientBtn" onClick={startNewPatient}><UserPlus size={18} /> Ajouter un patient</button>
      </aside>

      <main>
        <header>
          <div><h1>{view === 'patients' ? 'Mes patients' : view === 'dashboard' ? 'Patients graves' : view === 'messagerie' ? 'Messagerie & Bilans patients' : view === 'settings' ? 'Paramètres' : view === 'importExport' ? 'Importer / Exporter' : view === 'statistics' ? 'Statistiques' : 'Tableau de bord patient'}</h1><p>Anticipez les décompensations, priorisez les patients et personnalisez le suivi.</p></div>
          <div className="doctorBar">
            <div className="doctorChip">{doctor.photo_url ? <img src={doctor.photo_url} alt="Profil" /> : <User size={18} />}<div><b>{doctor.name}</b><small>{doctor.role || 'Cardiologue'}</small></div></div>
            <button className="ghostBtn noMargin" onClick={logout}>Déconnexion</button>
          </div>
        </header>

        {error && <div className="error">{error}</div>}

        {view === 'dashboard' && (
          <section className="panel">
            <h2>Patients avec probabilité grave</h2>
            <div className="patientTable">
              {criticalPatients.length === 0 && <p className="muted">Aucun patient grave enregistré pour ce médecin.</p>}
              {criticalPatients.map((p) => <PatientRow key={p.id} patient={p} onClick={() => openPatient(p)} />)}
            </div>
          </section>
        )}

        {view === 'patients' && (
          <section className="panel">
            <div className="sectionHeader"><h2>Base de données patients</h2><button className="primaryBtn" onClick={startNewPatient}><UserPlus size={16} /> Nouveau patient</button></div>
            <div className="patientTable">
              {patients.length === 0 && <p className="muted">Aucun patient enregistré.</p>}
              {patients.map((p) => <PatientRow key={p.id} patient={p} onClick={() => openPatient(p)} />)}
            </div>
          </section>
        )}

        {view === 'messagerie' && (
          <section className="panel">
            <h2><MessageSquare size={18}/> Conversations patients</h2>
            <p className="muted">Patients ayant envoyé un message ou un bilan. Cliquez pour répondre.</p>
            {conversations.length===0 && <p className="muted" style={{ marginTop:12 }}>Aucune conversation pour l'instant.</p>}
            <div style={{ display:'grid', gridTemplateColumns: selectedConv ? '320px 1fr' : '1fr', gap:16, marginTop:12 }}>
              <div style={{ display:'grid', gap:8, alignContent:'start' }}>
                {conversations.map((c)=>(
                  <button key={c.patient_auth_id} onClick={async()=>{
                    setSelectedConv(c);
                    // load latest messages
                    const res = await fetch(`${API_URL}/api/messages?patient_auth_id=${c.patient_auth_id}&doctor_id=${currentDoctorId}`);
                    const data = await res.json();
                    const bRes = await fetch(`${API_URL}/api/bilans?patient_auth_id=${c.patient_auth_id}&doctor_id=${currentDoctorId}`);
                    const bData = await bRes.json();
                    setSelectedConv({ ...c, messages: data.messages, bilans: bData.bilans });
                  }} style={{ textAlign:'left', background: selectedConv?.patient_auth_id===c.patient_auth_id ? '#eef2ff' : '#f8fafc', border:'1px solid #e5e7eb', borderRadius:12, padding:12, position:'relative' }}>
                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                      <div><b>{c.patient?.prenom || ''} {c.patient?.nom || 'Patient'} </b><small style={{ color:'#64748b' }}> — {c.patient?.email || ''}</small></div>
                      {c.unread_count>0 && <span style={{ background:'#ef4444', color:'white', borderRadius:999, padding:'2px 7px', fontSize:11, fontWeight:800 }}>1</span>}
                    </div>
                    <small className="muted">{c.last_message ? c.last_message.content.slice(0,60) : (c.bilans?.[0]?.filename || 'Bilan sans message')}</small><br/>
                    {c.bilans?.length>0 && <small style={{ color: c.unread_detail?.bilans>0 ? '#ef4444':'#4f46e5' }}><FileText size={12}/> {c.bilans.length} bilan(s) {c.unread_detail?.bilans>0 && `• ${c.unread_detail.bilans} non lu(s)`}</small>}
                  </button>
                ))}
              </div>
              {selectedConv && (
                <div style={{ display:'grid', gap:12 }}>
                  <div className="panel" style={{ background:'#f8fafc' }}>
                    <h3>{selectedConv.patient?.prenom} {selectedConv.patient?.nom} — {selectedConv.patient?.email}</h3>
                    <p className="muted">Né(e) {selectedConv.patient?.date_naissance || '--'} — {selectedConv.patient?.sexe || '--'}</p>
                    <div style={{ display:'grid', gap:6, maxHeight:260, overflowY:'auto', background:'white', padding:10, borderRadius:10, border:'1px solid #e5e7eb', marginTop:8 }}>
                      {(selectedConv.messages||[]).length===0 && <p className="muted">Pas de messages</p>}
                      {(selectedConv.messages||[]).map(m=>(
                        <div key={m.id} style={{ justifySelf: m.sender_role==='doctor'?'end':'start', background: m.sender_role==='doctor'?'#4f46e5':'#f1f5f9', color: m.sender_role==='doctor'?'white':'#1e293b', padding:'8px 10px', borderRadius:10, maxWidth:'82%', fontSize:12 }}>
                          {m.content}<br/><small style={{ opacity:.7 }}>{new Date(m.created_at).toLocaleString()}</small>
                        </div>
                      ))}
                    </div>
                    <div style={{ display:'flex', gap:8, marginTop:10 }}>
                      <input style={{ flex:1 }} placeholder="Répondre au patient..." value={doctorReply} onChange={e=>setDoctorReply(e.target.value)} onKeyDown={e=>e.key==='Enter' && sendDoctorMessage()} />
                      <button className="primaryBtn" onClick={sendDoctorMessage} disabled={loading}><Send size={16}/> Répondre</button>
                    </div>
                  </div>
                  <div className="panel">
                    <h3><FileText size={16}/> Bilans du patient</h3>
                    {(selectedConv.bilans||[]).length===0 && <p className="muted">Aucun bilan</p>}
                    {(selectedConv.bilans||[]).map(b=>(
                      <div key={b.id} style={{ border:'1px solid #e5e7eb', borderRadius:10, padding:10, marginTop:8, background:'#fff' }}>
                        <b>{b.filename}</b> <small className="muted">— {b.description || 'sans description'} — {new Date(b.created_at).toLocaleString()}</small><br/>
                        <small>Status: {b.status}</small>
                        <div style={{ display:'flex', gap:8, marginTop:6 }}>
                          <a href={`${API_URL}/api/bilans/download/${b.id}`} target="_blank" rel="noreferrer" className="ghostBtn noMargin" style={{ padding:'6px 10px', fontSize:12, textDecoration:'none' }}>Voir / Télécharger</a>
                        </div>
                        {b.reply ? <div style={{ marginTop:8, background:'#dcfce7', padding:8, borderRadius:8, fontSize:12 }}><b>Votre réponse:</b> {b.reply}</div> : (
                          <div style={{ display:'flex', gap:8, marginTop:8 }}>
                            <input style={{ flex:1 }} placeholder="Répondre au bilan..." value={bilanReply[b.id]||''} onChange={e=>setBilanReply(p=>({...p, [b.id]:e.target.value}))} />
                            <button className="secondaryBtn" style={{ marginTop:0 }} onClick={()=>replyBilan(b.id)}><Send size={14}/> Envoyer réponse</button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>
        )}

        {view === 'settings' && (
          <section className="panel settingsPanel">
            <h2>Profil cardiologue</h2>
            <div className="settingsGrid">
              <div className="profilePhotoBox">
                <div className="profilePreview">{settingsForm.photo_url ? <img src={settingsForm.photo_url} alt="Profil" /> : <User size={42} />}</div>
                <label className="fileBtn">
                  Changer la photo
                  <input type="file" accept="image/*" onChange={(e) => updateProfilePhoto(e.target.files?.[0])} />
                </label>
              </div>
              <div className="settingsForm">
                <label><span>Nom du cardiologue</span><input value={settingsForm.name} onChange={(e) => setSettingsForm((p) => ({ ...p, name: e.target.value }))} /></label>
                <label><span>Nouveau mot de passe</span><input type="password" placeholder="Laisser vide pour ne pas changer" value={settingsForm.password} onChange={(e) => setSettingsForm((p) => ({ ...p, password: e.target.value }))} /></label>
                <button className="primaryBtn" onClick={saveSettings} disabled={loading}>{loading ? <Loader2 className="spin" size={18} /> : <Settings size={18} />} Enregistrer les paramètres</button>
              </div>
            </div>
          </section>
        )}

        {view === 'importExport' && (
          <section className="panel importPanel">
            <h2>Importer / Exporter les patients</h2>
            <div className="importExportGrid">
              <div className="importBox">
                <Upload size={34} />
                <h3>Importer un fichier</h3>
                <p>Le fichier doit contenir : nom, prénom, date_naissance, sexe et les 16 variables du modèle.</p>
                <label className="fileBtn">
                  Choisir un fichier CSV ou Excel
                  <input type="file" accept=".csv,.xlsx,.xls" onChange={(e) => importPatients(e.target.files?.[0])} />
                </label>
                {importStatus && <div className="successMsg">{importStatus}</div>}
              </div>
              <div className="importBox">
                <Download size={34} />
                <h3>Exporter les patients</h3>
                <p>Exporter tous les patients et leurs consultations dans un fichier Excel.</p>
                <button className="primaryBtn" onClick={exportPatients}><Download size={18} /> Exporter Excel</button>
              </div>
            </div>
            <div className="columnsHint">
              <b>Colonnes attendues (19 vars modèle 100% - dose_sup120 dérivé auto) :</b>
              <code>nom, prenom, date_naissance, sexe, espace_PR, cause_valvulaire, PAD, PAS, OG, Uree, statine, ATCD_d_hospitalisation, IMC, IEC_dose, FQ_ECG_sortie, HTAP, TP, lymphocyte, ARM, QT_corrige, Glycemie_a_jeun, ProBNP, dose_de_lasilix</code>
              <b>Colonnes ordonnance optionnelles :</b>
              <code>lasilix, betabloquants, dose_BB_pourcentage, plavix_cardiocine, sintrome_aod, plavix, cardiocine100, sintrom, AOD, ARM, INH_SGLT2, Ivabradine</code>
            </div>
          </section>
        )}

        {view === 'statistics' && (
          <section className="panel statsPanel">
            <h2>Statistiques du cardiologue</h2>
            <div className="statsGrid">
              <div className="statCard"><span>Patients</span><strong>{patients.length}</strong></div>
              <div className="statCard"><span>Consultations</span><strong>{totalConsultations}</strong></div>
              <div className="statCard"><span>Patients graves</span><strong>{criticalPatients.length}</strong></div>
              <div className="statCard"><span>Score moyen</span><strong>{averageScore.toFixed(1)}%</strong></div>
            </div>
            <div className="riskDistribution">
              <h3>Répartition par niveau de risque</h3>
              {Object.entries(riskCounts).length === 0 && <p className="muted">Aucune donnée disponible.</p>}
              {Object.entries(riskCounts).map(([level, count]) => (
                <div className="riskLine" key={level}>
                  <span>{level}</span>
                  <div><i style={{ width: `${patients.length ? (count / patients.length) * 100 : 0}%`, background: riskColor(level) }} /></div>
                  <b>{count}</b>
                </div>
              ))}
            </div>
          </section>
        )}

        {(view === 'add' || view === 'detail') && (
          <>
            <section className="patientHero">
              <div className="avatar"><User size={34} /></div>
              <div className="patientIdentity">
                <h2>{activePatient.prenom || 'Prénom'} {activePatient.nom || 'Nom'} · Patient #{activePatient.id}</h2>
                <p>Age {calculateAge(identity.date_naissance)} ans · {identity.sexe || 'Sexe non renseigné'}</p>
                <span>{selectedPatient ? 'Historique disponible' : 'Nouveau patient'} · Consultation du jour</span>
              </div>
              <div className="heroMetric"><ShieldPlus size={22} /><span>BNP</span><b>{variables.ProBNP ?? '--'}</b></div>
              <div className="heroMetric"><Activity size={22} /><span>HTAP</span><b>{variables.HTAP ?? '--'}</b></div>
              <div className="heroMetric"><HeartPulse size={22} /><span>QTc</span><b>{variables.QT_corrige} ms</b></div>
              <div className="heroMetric"><CalendarClock size={22} /><span>Consult.</span><b>{selectedPatient?.consultations?.length || 0}</b></div>
            </section>

            <section className="dashboardGrid">
              <div className="panel scorePanel">
                <h2>Score IA - Risque de réhospitalisation à 3 mois</h2>
                {activeRisk ? <Gauge score={activeRisk.score_percent} level={activeRisk.risk_level} /> : <p>En attente...</p>}
              </div>
              <div className="panel factorsPanel">
                <h2>Facteurs majeurs contribuant au risque</h2>
                <div className="factorList">
                  {result ? result.major_factors.slice(0, 5).map((f) => <div className="factor" key={f.variable}><span>{f.variable}</span><b className={f.contribution >= 0 ? 'up' : 'down'}>{f.contribution >= 0 ? '+' : ''}{f.contribution.toFixed(3)}</b></div>) : <p className="muted">Calculer pour afficher les facteurs détaillés.</p>}
                </div>
              </div>
              <div className="sideStack">
                <div className="panel mortalityCard"><h2>Niveau de risque</h2><strong style={{ color: activeRisk ? riskColor(activeRisk.risk_level) : '#64748b' }}>{activeRisk ? activeRisk.risk_level : '--'}</strong><span>{activeRisk ? `${activeRisk.score_percent.toFixed(1)}% estimé` : 'En attente'}</span></div>
                <div className="panel alertsPanel"><h2>Alertes actives</h2>{result ? result.dashboard.alerts.slice(0, 3).map((a, i) => <div className={`alert ${a.priority}`} key={i}><AlertTriangle size={18} /><div><b>{a.title}</b><span>{a.detail}</span></div></div>) : <p className="muted">Calculer pour afficher les alertes.</p>}</div>
              </div>
            </section>

            <section className="panel formPanel">
              <h2>{selectedPatient ? 'Nouvelle consultation' : 'Ajouter un patient'}</h2>
              <div className="identityGrid">
                <label><span>Nom</span><input value={identity.nom} onChange={(e) => setIdentity((p) => ({ ...p, nom: e.target.value }))} /></label>
                <label><span>Prénom</span><input value={identity.prenom} onChange={(e) => setIdentity((p) => ({ ...p, prenom: e.target.value }))} /></label>
                <label><span>Date de naissance</span><input type="date" value={identity.date_naissance} onChange={(e) => setIdentity((p) => ({ ...p, date_naissance: e.target.value }))} /></label>
                <label><span>Sexe</span><select value={identity.sexe} onChange={(e) => setIdentity((p) => ({ ...p, sexe: e.target.value }))}><option value="">--</option><option value="M">Masculin</option><option value="F">Féminin</option></select></label>
              </div>
              <div className="formGrid">
                {fields.map(([key, label, unit]) => <label key={key}><span>{label}<small>{unit}</small></span><input type="number" step="any" value={variables[key]} onChange={(e) => setVariables((p) => ({ ...p, [key]: e.target.value === '' ? '' : Number(e.target.value) }))} /></label>)}
              </div>
              <button type="button" onClick={() => setShowOrdonnance((value) => !value)} className="ghostBtn ordonnanceToggle"><Stethoscope size={18} /> Ordonnance</button>
              {showOrdonnance && <Ordonnance variables={variables} setVariables={setVariables} />}
              <button onClick={() => submit(false)} disabled={loading} className="secondaryBtn">{loading ? <Loader2 className="spin" size={18} /> : <Activity size={18} />} Calculer le score et enregistrer</button>
              <button onClick={() => submit(true)} disabled={loading} className="primaryBtn actionGap">{loading ? <Loader2 className="spin" size={18} /> : <Stethoscope size={18} />} Générer rapport IA</button>
            </section>

            {result && <Recommendations result={result} />}
            {selectedPatient && <History patient={selectedPatient} onSelect={openConsultation} />}
          </>
        )}
      </main>
    </div>
  );
}

function PatientRow({ patient, onClick }) {
  const result = patient.last_result || {};
  return (
    <button className="patientRow" onClick={onClick}>
      <div><b>{patient.prenom || 'Prénom'} {patient.nom || 'Nom'}</b><span>Patient #{patient.id} · {calculateAge(patient.date_naissance)} ans · {patient.sexe || '--'}</span></div>
      <div><b style={{ color: riskColor(result.risk_level) }}>{result.score_percent ? `${result.score_percent}%` : '--'}</b><span>{result.risk_level || 'non calculé'}</span></div>
      <div><b>{patient.consultations?.length || 0}</b><span>consultations</span></div>
      <div><b>{patient.last_consultation?.slice(0, 10) || '--'}</b><span>dernière visite</span></div>
    </button>
  );
}

function Ordonnance({ variables, setVariables }) {
  const updateNumber = (key, value) => setVariables((previous) => ({ ...previous, [key]: value === '' ? '' : Number(value) }));
  return (
    <div className="ordonnancePanel">
      <div className="ordonnanceHeader">
        <div>
          <h3>Ordonnance de sortie</h3>
          <p>Renseigner les traitements trouvés dans data_ifrc et les doses associées.</p>
        </div>
      </div>
      <div className="ordonnanceGrid">
        {ordonnanceSelects.map(([key, label, options]) => (
          <label key={key}>
            <span>{label}</span>
            <select value={variables[key] ?? ''} onChange={(e) => updateNumber(key, e.target.value)}>
              {options.map(([value, text]) => <option value={value} key={`${key}-${value}`}>{text}</option>)}
            </select>
          </label>
        ))}
        <label>
          <span>Dose Lasilix<small>mg/j</small></span>
          <input type="number" step="any" disabled={Number(variables.lasilix) !== 1} value={variables.dose_de_lasilix ?? ''} onChange={(e) => updateNumber('dose_de_lasilix', e.target.value)} />
        </label>
        <label>
          <span>Dose betabloquant<small>%</small></span>
          <input type="number" step="any" disabled={Number(variables.betabloquants) !== 1} value={variables.dose_BB_pourcentage ?? ''} onChange={(e) => updateNumber('dose_BB_pourcentage', e.target.value)} />
        </label>
        <label>
          <span>Dose IEC<small>mg/j</small></span>
          <input type="number" step="any" value={variables.IEC_dose ?? ''} onChange={(e) => updateNumber('IEC_dose', e.target.value)} />
        </label>
      </div>
    </div>
  );
}

function Recommendations({ result }) {
  return (
    <section className="panel" id="recommendations">
      <h2>Recommandations personnalisées</h2>
      <div className="recommendationGrid">
        {result.dashboard.recommendation_cards.slice(0, 5).map((c, i) => <div className={`recCard priorite-${c.priority}`} key={i}><span className="priority">{c.priority}</span><h3>{c.category}</h3><p><b>Action:</b> {c.action}</p><p><b>Pourquoi:</b> {c.justification}</p></div>)}
      </div>
      {result.report_markdown && <div className="reportPanel"><h2>Rapport IA</h2><pre>{result.report_markdown}</pre></div>}
    </section>
  );
}

function History({ patient, onSelect }) {
  return (
    <section className="panel historyPanel">
      <h2>Historique du patient</h2>
      <div className="historyList">
        {[...(patient.consultations || [])].reverse().map((c, i) => (
          <button className="historyItem" key={`${c.date_consultation}-${i}`} onClick={() => onSelect(c)}>
            <div>
              <b>{c.date_consultation?.replace('T', ' ')}</b>
              <span>BNP {c.variables?.ProBNP ?? '--'} · HTAP {c.variables?.HTAP ?? '--'} · QTc {c.variables?.QT_corrige} ms</span>
              <em>Cliquer pour afficher score, alertes et recommandations</em>
            </div>
            <strong style={{ color: riskColor(c.resultat?.risk_level) }}>{c.resultat?.score_percent}% · {c.resultat?.risk_level}</strong>
          </button>
        ))}
      </div>
    </section>
  );
}

createRoot(document.getElementById('root')).render(<App />);
