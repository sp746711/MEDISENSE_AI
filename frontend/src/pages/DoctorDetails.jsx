import { useEffect, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import DoctorCard from '../components/DoctorCard';
import ErrorMessage from '../components/ErrorMessage';
import LoadingScreen from '../components/LoadingScreen';
import { getDoctor } from '../services/providerService';

export default function DoctorDetails() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [doctor, setDoctor] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;
    (async () => {
      try {
        const data = await getDoctor(id);
        if (!active) return;
        if (data.status === 'unavailable') {
          setError(data.message || 'Healthcare provider information is currently unavailable.');
        } else {
          setDoctor(data);
        }
      } catch (err) {
        if (active) setError(err.message);
      } finally {
        if (active) setLoading(false);
      }
    })();
    return () => {
      active = false;
    };
  }, [id]);

  if (loading) return <LoadingScreen message="Loading provider..." />;

  return (
    <div className="page narrow">
      <h1>Doctor Details</h1>
      <ErrorMessage message={error} onDismiss={() => setError('')} />
      {doctor ? <DoctorCard doctor={doctor} /> : null}

      <div style={{ marginTop: '20px', display: 'flex', gap: '12px' }}>
        {doctor ? (
          <button
            type="button"
            className="btn btn-primary"
            onClick={() =>
              navigate('/appointments/new', {
                state: {
                  doctorId: doctor.doctor_id,
                  doctorName: doctor.name,
                  doctorSpecialization: doctor.specialization,
                  doctorFacility: doctor.facility,
                },
              })
            }
          >
            Book Demo Appointment
          </button>
        ) : null}
        <button type="button" className="btn btn-secondary" onClick={() => navigate('/doctors')}>
          Back to Doctors
        </button>
      </div>

      <p className="disclaimer-inline" style={{ marginTop: '16px' }}>
        Appointment booking in MediSense AI is demonstration only. It creates an internal demonstration log and does not contact external healthcare providers.
      </p>
    </div>
  );
}
