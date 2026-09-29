import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import ProtectedRoute from './components/ProtectedRoute';
import { AuthProvider } from './context/AuthContext';
import { AssessmentProvider } from './context/AssessmentContext';
import { useAuth } from './hooks/useAuth';

import Login from './pages/Login';
import Signup from './pages/Signup';
import Dashboard from './pages/Dashboard';
import NewAssessment from './pages/NewAssessment';
import SymptomsInput from './pages/SymptomsInput';
import MedicalReportUpload from './pages/MedicalReportUpload';
import XrayUpload from './pages/XrayUpload';
import AssessmentProcessing from './pages/AssessmentProcessing';
import AssessmentResult from './pages/AssessmentResult';
import AssessmentDetails from './pages/AssessmentDetails';
import History from './pages/History';
import AIAssistant from './pages/AIAssistant';
import Doctors from './pages/Doctors';
import DoctorDetails from './pages/DoctorDetails';
import Facilities from './pages/Facilities';
import MedicalShops from './pages/MedicalShops';
import Appointments from './pages/Appointments';
import BookAppointment from './pages/BookAppointment';
import Profile from './pages/Profile';
import Settings from './pages/Settings';

function AppLayout() {
  const { isAuthenticated } = useAuth();
  return (
    <div className="app-shell">
      {isAuthenticated ? <Navbar /> : null}
      <main className="app-main">
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/signup" element={<Signup />} />

          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/assessment/new" element={<NewAssessment />} />
            <Route path="/assessment/symptoms" element={<SymptomsInput />} />
            <Route path="/assessment/report" element={<MedicalReportUpload />} />
            <Route path="/assessment/xray" element={<XrayUpload />} />
            <Route path="/assessment/processing" element={<AssessmentProcessing />} />
            <Route path="/assessment/:id" element={<AssessmentResult />} />
            <Route path="/assessment/:id/details" element={<AssessmentDetails />} />
            <Route path="/history" element={<History />} />
            <Route path="/assistant" element={<AIAssistant />} />
            <Route path="/doctors" element={<Doctors />} />
            <Route path="/doctors/:id" element={<DoctorDetails />} />
            <Route path="/facilities" element={<Facilities />} />
            <Route path="/medical-shops" element={<MedicalShops />} />
            <Route path="/appointments" element={<Appointments />} />
            <Route path="/appointments/new" element={<BookAppointment />} />
            <Route path="/profile" element={<Profile />} />
            <Route path="/settings" element={<Settings />} />
          </Route>

          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AssessmentProvider>
          <AppLayout />
        </AssessmentProvider>
      </AuthProvider>
    </BrowserRouter>
  );
}
