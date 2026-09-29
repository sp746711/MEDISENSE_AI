import { createContext, useCallback, useMemo, useState } from 'react';

export const AssessmentContext = createContext(null);

const STORAGE_KEY = 'medisense_assessment_draft';

function loadDraft() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    return raw
      ? JSON.parse(raw)
      : { assessmentId: null, inputTypes: [], stepIndex: 0 };
  } catch {
    return { assessmentId: null, inputTypes: [], stepIndex: 0 };
  }
}

export function AssessmentProvider({ children }) {
  const [draft, setDraft] = useState(loadDraft);

  const persist = useCallback((next) => {
    setDraft(next);
    sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  }, []);

  const setInputTypes = useCallback(
    (inputTypes) => persist({ ...draft, inputTypes }),
    [draft, persist],
  );

  const setAssessmentId = useCallback(
    (assessmentId) => persist({ ...draft, assessmentId }),
    [draft, persist],
  );

  const clearDraft = useCallback(() => {
    const empty = { assessmentId: null, inputTypes: [], stepIndex: 0 };
    persist(empty);
  }, [persist]);

  const value = useMemo(
    () => ({
      ...draft,
      setInputTypes,
      setAssessmentId,
      setDraft: persist,
      clearDraft,
    }),
    [draft, setInputTypes, setAssessmentId, persist, clearDraft],
  );

  return (
    <AssessmentContext.Provider value={value}>{children}</AssessmentContext.Provider>
  );
}
