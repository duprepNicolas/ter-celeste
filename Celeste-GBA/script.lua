previous_die = 0
function hasDie()
  if data.mort > previous_die then
    previous_die = data.mort
    return true
  else
    return false
  end
end

current_level = 0
function nextLevel()
    if data.nextLevel > current_level then
      return true
    else
      return false
    end
end

function done_Check()
    if hasDie() || nextLevel() then
        return true
    else
        return false
    end
end