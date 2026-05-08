// src/pages/PracticeAgent.tsx
// ─────────────────────────────────────────────────────────────────────
// Path B — Practice Agent page (Req 11.1, 11.3, 11.4, 11.5).
//
// Features:
//  • Topic input and difficulty dropdown (easy / medium / hard).
//  • "Start Practice" button calls practiceStart and renders the MCQ.
//  • Four radio-button answer options; Submit is disabled until one is
//    selected (Req 11.3).
//  • Calls practiceAnswer on submit and renders ✓ Correct / ✗ Incorrect
//    feedback.
//  • When the response includes an explanation (Deep_Dive_Tool output),
//    renders it in a bg-yellow-50 border-l-4 border-yellow-400 panel
//    with a 📚 icon to visually distinguish it from standard feedback
//    (Req 11.3).
//  • Shows LoadingSpinner while any API call is in flight (Req 11.4).
//  • Shows ErrorBanner on non-2xx API errors (Req 11.5).
// ─────────────────────────────────────────────────────────────────────

import { useState } from 'react'
import {
  practiceStart,
  practiceAnswer,
  type PracticeStartResponse,
  type PracticeAnswerResponse,
} from '../api/client'
import ErrorBanner from '../components/ErrorBanner'
import LoadingSpinner from '../components/LoadingSpinner'

// ── Types ─────────────────────────────────────────────────────────────

type Difficulty = 'easy' | 'medium' | 'hard'

/** The phase the UI is currently in. */
type Phase =
  | 'setup'       // Topic + difficulty form, before any question is loaded
  | 'question'    // MCQ is displayed, waiting for the student to answer
  | 'feedback'    // Answer has been submitted; showing outcome + optional explanation

// ── Component ─────────────────────────────────────────────────────────

export default function PracticeAgent() {
  // ── Setup form state ────────────────────────────────────────────────
  const [topic, setTopic] = useState('')
  const [difficulty, setDifficulty] = useState<Difficulty>('medium')

  // ── Session / question state ────────────────────────────────────────
  const [phase, setPhase] = useState<Phase>('setup')
  const [currentQuestion, setCurrentQuestion] = useState<PracticeStartResponse | null>(null)
  const [selectedOption, setSelectedOption] = useState<string>('')

  // ── Feedback state ──────────────────────────────────────────────────
  const [feedback, setFeedback] = useState<PracticeAnswerResponse | null>(null)

  // ── API call state ──────────────────────────────────────────────────
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // ── Handlers ────────────────────────────────────────────────────────

  /** Called when the student clicks "Start Practice". */
  async function handleStart() {
    const trimmedTopic = topic.trim()
    if (!trimmedTopic || isLoading) return

    setError(null)
    setIsLoading(true)
    setSelectedOption('')
    setFeedback(null)

    try {
      const response = await practiceStart({ topic: trimmedTopic, difficulty_level: difficulty })
      setCurrentQuestion(response)
      setPhase('question')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start practice session.')
    } finally {
      setIsLoading(false)
    }
  }

  /** Called when the student clicks "Submit Answer". */
  async function handleSubmitAnswer() {
    if (!currentQuestion || !selectedOption || isLoading) return

    setError(null)
    setIsLoading(true)

    try {
      const response = await practiceAnswer({
        session_id: currentQuestion.session_id,
        question_id: currentQuestion.question_id,
        selected_option: selectedOption,
      })
      setFeedback(response)
      setPhase('feedback')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to submit answer.')
    } finally {
      setIsLoading(false)
    }
  }

  /** Reset to the setup form so the student can start a new question. */
  function handleReset() {
    setPhase('setup')
    setCurrentQuestion(null)
    setSelectedOption('')
    setFeedback(null)
    setError(null)
  }

  /** Start a new question on the same topic / difficulty without going back to setup. */
  async function handleNextQuestion() {
    if (!topic.trim() || isLoading) return

    setError(null)
    setIsLoading(true)
    setSelectedOption('')
    setFeedback(null)
    setPhase('setup') // briefly reset so stale question isn't visible

    try {
      const response = await practiceStart({ topic: topic.trim(), difficulty_level: difficulty })
      setCurrentQuestion(response)
      setPhase('question')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load next question.')
      setPhase('setup')
    } finally {
      setIsLoading(false)
    }
  }

  // ── Render ──────────────────────────────────────────────────────────

  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto">
      {/* ── Page header ─────────────────────────────────────────────── */}
      <div>
        <h2 className="text-xl font-semibold text-gray-800">Practice Agent</h2>
        <p className="mt-1 text-sm text-gray-500">
          Choose a topic and difficulty, then answer multiple-choice questions. Incorrect
          answers trigger a detailed concept explanation from the Deep-Dive tool.
        </p>
      </div>

      {/* ── Error banner ─────────────────────────────────────────────── */}
      <ErrorBanner error={error} onDismiss={() => setError(null)} />

      {/* ── Setup form ───────────────────────────────────────────────── */}
      {/*
        Always visible so the student can change topic/difficulty at any time.
        Inputs are disabled while a question is active to prevent accidental
        changes mid-session.
      */}
      <SetupForm
        topic={topic}
        difficulty={difficulty}
        isLoading={isLoading}
        isDisabled={phase !== 'setup'}
        onTopicChange={setTopic}
        onDifficultyChange={setDifficulty}
        onStart={handleStart}
        onReset={phase !== 'setup' ? handleReset : undefined}
      />

      {/* ── Loading spinner ───────────────────────────────────────────── */}
      <LoadingSpinner isLoading={isLoading} />

      {/* ── MCQ card ─────────────────────────────────────────────────── */}
      {currentQuestion && phase !== 'setup' && (
        <MCQCard
          question={currentQuestion}
          selectedOption={selectedOption}
          isSubmitted={phase === 'feedback'}
          isLoading={isLoading}
          onSelectOption={setSelectedOption}
          onSubmit={handleSubmitAnswer}
        />
      )}

      {/* ── Feedback panel ───────────────────────────────────────────── */}
      {feedback && phase === 'feedback' && (
        <FeedbackPanel
          feedback={feedback}
          onNextQuestion={handleNextQuestion}
          onReset={handleReset}
          isLoading={isLoading}
        />
      )}
    </div>
  )
}

// ── SetupForm sub-component ───────────────────────────────────────────

interface SetupFormProps {
  topic: string
  difficulty: Difficulty
  isLoading: boolean
  isDisabled: boolean
  onTopicChange: (v: string) => void
  onDifficultyChange: (v: Difficulty) => void
  onStart: () => void
  onReset?: () => void
}

/**
 * Topic input, difficulty dropdown, and Start Practice / Change Topic buttons.
 * When isDisabled is true the inputs are read-only and a "Change Topic" button
 * is shown instead of "Start Practice".
 */
function SetupForm({
  topic,
  difficulty,
  isLoading,
  isDisabled,
  onTopicChange,
  onDifficultyChange,
  onStart,
  onReset,
}: SetupFormProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      <h3 className="mb-4 text-sm font-semibold text-gray-700 uppercase tracking-wide">
        Session Setup
      </h3>

      <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
        {/* Topic input */}
        <div className="flex-1">
          <label htmlFor="practice-topic" className="mb-1 block text-sm font-medium text-gray-700">
            Topic
          </label>
          <input
            id="practice-topic"
            type="text"
            value={topic}
            onChange={(e) => onTopicChange(e.target.value)}
            placeholder="e.g. quadratic equations"
            disabled={isDisabled || isLoading}
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-400"
            aria-label="Practice topic"
          />
        </div>

        {/* Difficulty dropdown */}
        <div className="sm:w-36">
          <label
            htmlFor="practice-difficulty"
            className="mb-1 block text-sm font-medium text-gray-700"
          >
            Difficulty
          </label>
          <select
            id="practice-difficulty"
            value={difficulty}
            onChange={(e) => onDifficultyChange(e.target.value as Difficulty)}
            disabled={isDisabled || isLoading}
            className="w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-100 disabled:text-gray-400"
            aria-label="Difficulty level"
          >
            <option value="easy">Easy</option>
            <option value="medium">Medium</option>
            <option value="hard">Hard</option>
          </select>
        </div>

        {/* Action button */}
        {isDisabled ? (
          // "Change Topic" lets the student go back to setup without losing the
          // current question display — onReset handles the state transition.
          <button
            onClick={onReset}
            disabled={isLoading}
            className="shrink-0 rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Change Topic
          </button>
        ) : (
          <button
            onClick={onStart}
            disabled={isLoading || !topic.trim()}
            className="shrink-0 rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Start Practice
          </button>
        )}
      </div>
    </div>
  )
}

// ── MCQCard sub-component ─────────────────────────────────────────────

interface MCQCardProps {
  question: PracticeStartResponse
  selectedOption: string
  isSubmitted: boolean
  isLoading: boolean
  onSelectOption: (option: string) => void
  onSubmit: () => void
}

/**
 * Renders the multiple-choice question with four radio-button options.
 *
 * The Submit button is disabled until an option is selected (Req 11.3).
 * After submission (isSubmitted=true) the radio buttons become read-only
 * so the student can see which option they chose while reading feedback.
 */
function MCQCard({
  question,
  selectedOption,
  isSubmitted,
  isLoading,
  onSelectOption,
  onSubmit,
}: MCQCardProps) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
      {/* Question stem */}
      <p className="mb-5 text-base font-medium text-gray-900 leading-relaxed">
        {question.question_text}
      </p>

      {/* Answer options — four radio buttons */}
      <fieldset className="mb-5" aria-label="Answer options">
        <legend className="sr-only">Select your answer</legend>
        <div className="flex flex-col gap-3">
          {question.options.map((optionText, index) => {
            // Option letters: A, B, C, D
            const letter = String.fromCharCode(65 + index)
            const optionId = `option-${letter}`
            const isSelected = selectedOption === letter

            return (
              <label
                key={letter}
                htmlFor={optionId}
                className={`flex cursor-pointer items-start gap-3 rounded-lg border px-4 py-3 text-sm transition-colors ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 text-blue-900'
                    : 'border-gray-200 bg-gray-50 text-gray-800 hover:border-gray-300 hover:bg-gray-100'
                } ${isSubmitted || isLoading ? 'cursor-default' : ''}`}
              >
                <input
                  id={optionId}
                  type="radio"
                  name="mcq-option"
                  value={letter}
                  checked={isSelected}
                  onChange={() => !isSubmitted && onSelectOption(letter)}
                  disabled={isSubmitted || isLoading}
                  className="mt-0.5 h-4 w-4 shrink-0 accent-blue-600"
                  aria-label={`Option ${letter}: ${optionText}`}
                />
                <span>
                  <span className="font-semibold">{letter}.</span> {optionText}
                </span>
              </label>
            )
          })}
        </div>
      </fieldset>

      {/* Submit button — disabled until an option is selected (Req 11.3) */}
      {!isSubmitted && (
        <button
          onClick={onSubmit}
          disabled={!selectedOption || isLoading}
          className="rounded-md bg-blue-600 px-5 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
          aria-label="Submit answer"
        >
          Submit Answer
        </button>
      )}
    </div>
  )
}

// ── FeedbackPanel sub-component ───────────────────────────────────────

interface FeedbackPanelProps {
  feedback: PracticeAnswerResponse
  isLoading: boolean
  onNextQuestion: () => void
  onReset: () => void
}

/**
 * Renders the outcome feedback after the student submits an answer.
 *
 * Correct answer:   ✓ Correct — green banner.
 * Incorrect answer: ✗ Incorrect — red banner.
 *
 * When an explanation is present (Deep_Dive_Tool output on incorrect answers),
 * it is rendered in a visually distinct yellow panel with a 📚 icon to
 * distinguish it from standard feedback (Req 11.3).
 */
function FeedbackPanel({ feedback, isLoading, onNextQuestion, onReset }: FeedbackPanelProps) {
  const isCorrect = feedback.outcome === 'correct'

  return (
    <div className="flex flex-col gap-4">
      {/* ── Outcome banner ─────────────────────────────────────────── */}
      <div
        className={`flex items-center gap-3 rounded-lg border px-4 py-3 text-sm font-semibold ${
          isCorrect
            ? 'border-green-300 bg-green-50 text-green-800'
            : 'border-red-300 bg-red-50 text-red-800'
        }`}
        role="status"
        aria-live="polite"
      >
        <span className="text-lg" aria-hidden="true">
          {isCorrect ? '✓' : '✗'}
        </span>
        <span>{isCorrect ? 'Correct! Well done.' : 'Incorrect. Review the explanation below.'}</span>
      </div>

      {/* ── Deep-Dive explanation panel ────────────────────────────── */}
      {/*
        Rendered only when the backend returns an explanation, which happens
        when the student answers incorrectly and the Deep_Dive_Tool is invoked.
        The yellow left-border panel visually distinguishes this educational
        content from the standard correct/incorrect feedback (Req 11.3).
      */}
      {feedback.explanation && (
        <div
          className="bg-yellow-50 border-l-4 border-yellow-400 rounded-r-lg px-5 py-4"
          role="complementary"
          aria-label="Concept explanation"
        >
          {/* Header row with 📚 icon */}
          <div className="flex items-center gap-2 mb-3">
            <span className="text-xl" aria-hidden="true">📚</span>
            <h4 className="text-sm font-semibold text-yellow-900 uppercase tracking-wide">
              Concept Explanation
            </h4>
          </div>

          {/* Explanation text — preserve whitespace and line breaks from LLM output */}
          <p className="whitespace-pre-wrap text-sm text-yellow-900 leading-relaxed">
            {feedback.explanation}
          </p>
        </div>
      )}

      {/* ── Navigation buttons ──────────────────────────────────────── */}
      <div className="flex gap-3">
        <button
          onClick={onNextQuestion}
          disabled={isLoading}
          className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Next Question
        </button>
        <button
          onClick={onReset}
          disabled={isLoading}
          className="rounded-md border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-1 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Change Topic
        </button>
      </div>
    </div>
  )
}
