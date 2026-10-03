import { createContext, useCallback, useMemo, useState } from 'react';

export const AssessmentContext = createContext(null);

const STORAGE_KEY = 'medisense_assessment_draft';

function loadDraft() {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return { assessmentId: null, inputTypes: [], stepIndex: 0 };
    const parsed = JSON.parse(raw);
    return {
      assessmentId: parsed?.assessmentId || null,
      inputTypes: Array.isArray(parsed?.inputTypes) ? parsed.inputTypes : [],
      stepIndex: typeof parsed?.stepIndex === 'number' ? parsed.stepIndex : 0,
    };
  } catch {
    return { assessmentId: null, inputTypes: [], stepIndex: 0 };
  }
}

export function AssessmentProvider({ children }) {
  const [draft, setDraftState] = useState(loadDraft);

  // Single functional persistence updater that guarantees atomic merging against latest state
  const setDraft = useCallback((updater) => {
    setDraftState((prev) => {
      const next = typeof updater === 'function' ? updater(prev) : { ...prev, ...updater };
      try {
        sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      } catch (e) {
        console.error('Failed to save assessment draft to sessionStorage', e);
      }
      return next;
    });
  }, []);

  const setInputTypes = useCallback(
    (inputTypes) => {
      setDraft((prev) => ({
        ...prev,
        inputTypes: Array.isArray(inputTypes) ? inputTypes : [],
      }));
    },
    [setDraft],
  );

  const setAssessmentId = useCallback(
    (assessmentId) => {
      setDraft((prev) => ({
        ...prev,
        assessmentId,
      }));
    },
    [setDraft],
  );

  const clearDraft = useCallback(() => {
    const empty = { assessmentId: null, inputTypes: [], stepIndex: 0 };
    setDraftState(empty);
    try {
      sessionStorage.removeItem(STORAGE_KEY);
    } catch (e) {
      console.error('Failed to clear assessment draft from sessionStorage', e);
    }
  }, []);

  const value = useMemo(
    () => ({
      assessmentId: draft.assessmentId,
      inputTypes: draft.inputTypes,
      stepIndex: draft.stepIndex,
      setInputTypes,
      setAssessmentId,
      setDraft,
      clearDraft,
    }),
    [draft.assessmentId, draft.inputTypes, draft.stepIndex, setInputTypes, setAssessmentId, setDraft, clearDraft],
  );

  return (
    <AssessmentContext.Provider value={value}>{children}</AssessmentContext.Provider>
  );
}
