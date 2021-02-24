heighest = 17088
function isHeighest()
    if data.height < heighest then
        heighest = data.heighest
        return 1
    else
        return 0
    end
end

strawberry = 0
function collectStrawberry()
    if data.strawberry != strawberry then
        strawberry = data.strawberry
        return 1
    else
        return 0
    end
end

key = 0
function collectKeys()
    if data.key > key then
        key = data.key
        return 1
    else
        return 0
    end
end

death = 0
function isDead()
    id data.death > death then
        death = data.death
        return -1
    else
        return 0
    end
end