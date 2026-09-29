import { useContext } from 'react';
import { AssessmentContext } from '../context/AssessmentContext';

export function useAssessment() {
  const ctx = useContext(AssessmentContext);
  if (!ctx) throw new Error('useAssessment must be used within AssessmentProvider');
  return ctx;
}
